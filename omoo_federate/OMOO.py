"""
Basic State Estimation Federate

Uses weighted least squares to estimate the voltage angles.

First `call_h` calculates the residual from the voltage magnitude and angle,
and `call_H` calculates a jacobian. Then `scipy.optimize.least_squares`
is used to solve.
"""

import cmath
import warnings
import logging
import helics as h
import json
import numpy as np
import time
import pandas as pd
from pydantic import BaseModel
from enum import Enum
from typing import List, Optional, Union
from datetime import datetime
from oedisi.types.data_types import (
    AdmittanceMatrix,
    EquipmentNodeArray,
    MeasurementArray,
    Topology,
    Complex,
    VoltagesReal,
    VoltagesImaginary,
    Injection,
    CommandList,
    PowersReal,
    PowersImaginary,
    AdmittanceSparse,
    Command,
)
from scipy.sparse import csc_matrix, coo_matrix, diags, vstack, hstack
from scipy.sparse.linalg import svds, inv
import xarray as xr

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.DEBUG)


def eqarray_to_xarray(eq: EquipmentNodeArray):
    return xr.DataArray(
        eq.values,
        dims=("eqnode",),
        coords={
            "equipment_ids": ("eqnode", eq.equipment_ids),
            "ids": ("eqnode", eq.ids),
        },
    )


def measurement_to_xarray(eq: MeasurementArray):
    return xr.DataArray(eq.values, coords={"ids": eq.ids})


def matrix_to_numpy(admittance: List[List[Complex]]):
    "Convert list of list of our Complex type into a numpy matrix"
    return np.array([[x[0] + 1j * x[1] for x in row] for row in admittance])


def get_indices(topology, measurement):
    "Get list of indices in the topology for each index of the input measurement"
    inv_map = {v: i for i, v in enumerate(topology.base_voltage_magnitudes.ids)}
    return [inv_map[v] for v in measurement.ids]


def get_y(admittance: Union[AdmittanceMatrix, AdmittanceSparse], ids: List[str]):
    if type(admittance) == AdmittanceMatrix:
        assert ids == admittance.ids
        return matrix_to_numpy(admittance.admittance_matrix)
    elif type(admittance) == AdmittanceSparse:
        node_map = {name: i for (i, name) in enumerate(ids)}
        return coo_matrix(
            (
                [v[0] + 1j * v[1] for v in admittance.admittance_list],
                (
                    [node_map[r] for r in admittance.from_equipment],
                    [node_map[c] for c in admittance.to_equipment],
                ),
            )
        ).toarray()


def primal_dual(
    dual_update_v, muk_last, lambdak_last, epsilon, alpha, uni_Vmax, uni_Vmin, V_k
):
    """
    update dual variables
    lk >> (uni_Vmin - V_k) >> uni_Vmin - V_k <= 0
    mk >> (V_k - uni_Vmax) >> V_k - uni_Vmax <=0
    """
    ratio_primal_dual = 1
    if dual_update_v == ratio_primal_dual:
        alpha = alpha * 10  # Times 20 to speed up, may lose some optimality
        lk = lambdak_last + alpha * (uni_Vmin - V_k - epsilon * lambdak_last)
        # lk = (uni_Vmin - V_k)
        lambdak_cur = np.maximum(0, lk)

        mk = muk_last + alpha * (V_k - uni_Vmax - epsilon * muk_last)
        # mk = (V_k - uni_Vmax)
        muk_cur = np.maximum(0, mk)
        dual_update_v = 0
    else:
        # use outdated dual variables
        lambdak_cur = lambdak_last
        muk_cur = muk_last
        dual_update_v = dual_update_v + 1

    return dual_update_v, lambdak_cur, muk_cur


def Proj_inverter(xt, yt, Ux, Sx):
    """
    xt, yt: real, imag parts of PV power
    Ux = max available power currently
    Sx = max capacity power
    """
    # Ux = max available power currently
    # Sx = max capacity power
    if Ux - Sx > 0:  # If avaialble power is larger than capacity, use capacity
        Ux = Sx
    qx = np.sqrt(Sx**2 - Ux**2)
    theta = np.arcsin(qx / Sx)

    if xt < 0:
        warnings.warn("real power is negative")
        print(xt)

    try:
        theta_t = np.arcsin(yt / np.sqrt(xt**2 + yt**2 + 1e-8))
    except:
        theta_t = 0

    if xt**2 + yt**2 <= Sx**2 and xt <= Ux:
        # No violation
        x2 = xt
        y2 = yt

    else:  # theta_t is current p, q angle
        if abs(theta_t) > theta:
            x2 = xt * Sx / cmath.sqrt((xt**2 + yt**2))
            y2 = yt * Sx / cmath.sqrt((xt**2 + yt**2))

        if abs(theta_t) <= theta:
            if np.abs(yt) > cmath.sqrt(Sx**2 - Ux**2):
                x2 = Ux
                y2 = cmath.sqrt(Sx**2 - Ux**2)
            else:
                x2 = Ux
                y2 = yt
    if x2.real < 0:
        x2 = 0.0
    # if y2.real < 0:
    #     y2 = 0.
    return x2.real, y2.real


def cost_fun(
    Q_set_v,
    Qk_v,
    Qk_v_last,
    pv_ii,
    muk_last,
    lambdak_last,
    P_set_v,
    Pk_v,
    Pck_v,
    pv_avail_v,
    pv_capacity_v,
    H,
    G,
    nu,
    alpha,
    tot_cost_v,
):
    alph = 0.1
    cost_Q = 1 * alph
    cost_P = 1 * alph
    p_dev = 1 * (1 - alph)
    q_dev = 1 * (1 - alph)

    # f1 = q_dev * (Q_set_v - Qk_v)^2. q_dev-penalty; (Qk_v -Q_set_v)-deviation from set point.
    q_set_dev = q_dev * 2 * (-Q_set_v + Qk_v)
    Qk_v_last = Qk_v
    # f2 = cost_Q * Qk_v_last^2. cost_Q-penalty; Qk_v_last^2-penalize any reactive generation
    q_cost = 2 * cost_Q * Qk_v_last
    # gradient of f1, f2; lk * (uni_Vmin - (G @ p + H @ q)) + mk * ((G @ p + H @ q) - uni_max)
    dLq_pp = q_cost + q_set_dev + (H[:, pv_ii]) @ (muk_last - lambdak_last)  # + nu*Qk_v
    # dLq_pp = (H[:,pv_ii])@(muk_last - lambdak_last)
    uqk = Qk_v - alpha * dLq_pp  # gradient descent

    uqk = np.maximum(uqk, -0.2 * pv_capacity_v)
    uqk = np.minimum(uqk, 0.2 * pv_capacity_v)
    if pv_avail_v > 0:
        uqk = np.maximum(-1 / np.sqrt(2) * pv_avail_v, uqk)
        uqk = np.minimum(1 / np.sqrt(2) * pv_avail_v, uqk)

    # f1= p_dev * (P_set_v - Pk_v) ^ 2; Looks like curtailment since P_set_v is the availability
    p_set_dev = -p_dev * 2 * (P_set_v - Pk_v)
    # f2 = cost_P * Pck_v^2
    p_cost = 2 * cost_P * Pck_v
    # gradient of f1, f2; lk * (uni_Vmin - (G @ p + H @ q)) + mk * ((G @ p + H @ q) - uni_max)

    dLp_pp = p_cost + p_set_dev + (G[:, pv_ii]) @ (muk_last - lambdak_last)  # + nu*Pk_v
    # print(f"p_cost, p_set_dev {p_cost}, {p_set_dev}, {(G[:, pv_ii])@(muk_last - lambdak_last)}")
    upk = Pk_v - alpha * dLp_pp
    upk = np.maximum(0, upk)
    upk = np.minimum(upk, pv_avail_v)
    # upk = pv_avail_v
    # P, Q = Proj_inverter(upk, uqk, pv_avail_v, pv_capacity_v)

    P, Q = upk, uqk
    # print(dLp_pp, np.sum(muk_last[50]))
    Pck_v = pv_avail_v - P  # Curtailment
    # cost = P curtailment + Q generation + deviation for P and Q
    tot_cost_v += (
        cost_P * Pck_v**2
        + cost_Q * Q**2
        + q_dev * (Q_set_v - Qk_v) ** 2
        + p_dev * (P_set_v - Pk_v) ** 2
    )
    return P, Q, Pck_v, tot_cost_v, dLp_pp, dLq_pp


def pv_cost(
    G,
    H,
    N_node,
    Pk_last,
    Qk_last,
    Sbase,
    alpha,
    pv_index,
    pv_frame,
    lambdak_last,
    muk_last,
    nu,
    pv_set_point_real,
    pv_set_point_imag,
):
    tot_cost_v = 0.0
    Pk_current = np.zeros(len(pv_index))
    Qk_current = np.zeros(len(pv_index))
    Pck_last = np.zeros(len(pv_index))
    dLp_pp_list = np.zeros(N_node)
    dLq_pp_list = np.zeros(N_node)
    for pp, pv_ii in enumerate(pv_index):
        Pck_v = 0.0

        pv_avail_v = (
            pv_frame.iloc[pp]["avai"] / Sbase
        )  # available generation for this pv
        P_set_v = (
            pv_frame.iloc[pp]["avai"] - pv_set_point_real[pp]
        ) / Sbase  # deviation from set point
        Q_set_v = pv_set_point_imag[pp] / Sbase

        Pk_v = Pk_last[pp]
        Qk_v = Qk_last[pp]
        pv_capacity_v = pv_frame.iloc[pp]["kVarRated"] / Sbase
        P, Q, Pck_v, tot_cost_vv, dLp_pp, dLq_pp = cost_fun(
            Q_set_v,
            Qk_v,
            Qk_v,
            pv_ii,
            muk_last,
            lambdak_last,
            P_set_v,
            Pk_v,
            Pck_v,
            pv_avail_v,
            pv_capacity_v,
            H,
            G,
            nu,
            alpha,
            tot_cost_v,
        )
        # print(P, pv_avail_v)
        dLp_pp_list[pv_ii] = dLp_pp
        dLq_pp_list[pv_ii] = dLq_pp
        Pk_current[pp] = P
        Qk_current[pp] = Q
        Pck_last[pp] = Pck_v
        tot_cost_v = tot_cost_vv
    return Pk_current, Qk_current, dLp_pp_list, dLq_pp_list, tot_cost_v


class UnitSystem(str, Enum):
    SI = "SI"
    PER_UNIT = "PER_UNIT"


class OMOOParameters(BaseModel):
    Vmax: float = 1.05  # + 0.005  # Upper limit
    Vmin_act: float = 0.95  # + 0.005
    Vmin: float = 0.95  # + 0.005 + 0.002  # Lower limit\
    # Linearized equation is overestimating. So increase the lower limit by 0.005.
    # The problem is not solved to the optimal, so increase another 0.002.
    alpha: float = 0.5  #  learning rate for dual and primal
    epsilon: float = 1e-8  # decay of duals
    nu: float = 0.016  # A trivial penalty factor
    ratio_t_k: int = 1000
    units: UnitSystem = UnitSystem.PER_UNIT
    base_power: Optional[float] = 100.0


def getLinearModel(YLL, YL0, V0):
    """
    Calculates components in equation (3)
    [Decentralized Low-Rank State Estimation for Power Distribution Systems]
    ----
    Arguments:
        YLL, YL0: A part of Y matrix. Defined above equation (4)
        V0: True voltages of slack buses.
    Returns:
        w, w_mag, My, Ky: w, |w|, N, K in (3)
    """
    #
    Nnode = YLL.shape[0]
    YLLi = inv(YLL)  # quicker than spsolve(YLL_s, eye(YLL_s.shape[0])) by testing
    w = -((YLLi @ YL0) @ V0)
    w_mag = np.abs(w)
    Vk = w
    Vk_mag = np.abs(Vk)
    prody = YLLi @ inv(diags(np.conjugate(Vk).squeeze(), format="csc"))
    My = hstack([prody, -1j * prody])

    Ky = (
        inv(diags(np.conjugate(Vk_mag).squeeze(), format="csc"))
        @ (diags(np.conjugate(Vk).squeeze(), format="csc") @ My).real
    )
    Ky1, Ky2 = Ky[:, :Nnode], Ky[:, Nnode:]
    # Ky1, Ky2 are converted to dense, since they are not sparse
    return Ky1.toarray(), Ky2.toarray(), w_mag.reshape(-1)


class OMOO:
    def __init__(
        self,
        parameters: OMOOParameters,
        topology: Topology,
        slackbus_indices,
        V0,
        pv_frame,
        YLL,
        YL0,
        G,
        H,
        w_mag,
    ):
        self.parameters = parameters
        self.topology = topology
        self.num_node = len(topology.base_voltage_magnitudes.ids)
        self.base_voltages = np.array(topology.base_voltage_magnitudes.values)
        self.slack_bus = slackbus_indices
        self.V0 = (V0 / self.base_voltages[slackbus_indices]).reshape(3, -1)

        if self.parameters.units == UnitSystem.PER_UNIT:
            base_power = 100
            if self.parameters.base_power != None:
                base_power = self.parameters.base_power
        self.YLL = YLL
        self.YL0 = YL0
        self.base_power = base_power
        self.G, self.H, self.w_mag = G, H, w_mag
        self.pv_frame = pv_frame
        self.pv_index = pv_frame["index"].values.tolist()

    def opf_run(self, V, P_wopv, Q_wopv):
        if self.parameters.units == UnitSystem.SI:
            vmagTrue = np.array(V.values)
        elif self.parameters.units == UnitSystem.PER_UNIT:
            vmagTrue = np.array(V.values) / self.base_voltages
            P_wopv = -np.array(P_wopv.values) / self.base_power
            Q_wopv = -np.array(Q_wopv.values) / self.base_power
        else:
            raise Exception(f"Unit system {parameters.units} not supported")
        P_wopv, Q_wopv = (
            np.delete(P_wopv, self.slack_bus),
            np.delete(Q_wopv, self.slack_bus),
        )
        # Initial P and Q setpoint
        P_0, Q_0 = (
            self.pv_frame["avai"].values / self.base_power,
            np.zeros(len(self.pv_frame)),
        )
        Ppv, Qpv = np.zeros(self.num_node), np.zeros(self.num_node)
        for pp, pv_ii in enumerate(self.pv_index):
            Ppv[pv_ii] = P_0[pp]
            Qpv[pv_ii] = Q_0[pp]
        Ppv, Qpv = np.delete(Ppv, self.slack_bus), np.delete(Qpv, self.slack_bus)
        Vk_wopv = self.w_mag + self.G @ P_wopv + self.H @ Q_wopv
        V_hat = Vk_wopv + self.G @ Ppv + self.H @ Qpv
        diff = np.abs(V_hat - np.delete(vmagTrue, self.slack_bus))
        logger.debug(f"maximum diff is {np.max(diff)}")
        logger.debug(f"maximum V_hat is {np.max(V_hat)}")
        ind = (
            np.where(vmagTrue > self.parameters.Vmax)[0].tolist()
            + np.where(vmagTrue < self.parameters.Vmin_act)[0].tolist()
        )
        if len(ind) == 0:
            P_0, Q_0 = P_0 * self.base_power, Q_0 * self.base_power
            set_power = False
            logger.debug("Skip this step since no violation")
            logger.debug(f"minimum V_hat is {np.min(V_hat)}")
            logger.debug(f"maximum V_hat is {np.max(V_hat)}")
            return P_0, Q_0, set_power, V_hat
        # logger.debug('ind')
        # logger.debug(ind)
        else:
            V_k = np.delete(vmagTrue, self.slack_bus)
            N_node, N_pv = len(V_k), len(self.pv_frame)
            dual_update_v = 1
            muk_last, lambdak_last = np.zeros(N_node), np.zeros(N_node)
            Pk_last, Qk_last = P_0, Q_0 * 0.0
            pv_set_point_real, pv_set_point_imag = np.zeros(N_pv), np.zeros(N_pv)
            uni_Vmax = np.ones(N_node) * self.parameters.Vmax
            uni_Vmin = np.ones(N_node) * self.parameters.Vmin

            for jj in range(self.parameters.ratio_t_k):
                # Updating P, Q
                Pk_last, Qk_last, _, _, _ = pv_cost(
                    self.G,
                    self.H,
                    N_node,
                    Pk_last,
                    Qk_last,
                    self.parameters.base_power,
                    self.parameters.alpha,
                    self.pv_index,
                    self.pv_frame,
                    lambdak_last,
                    muk_last,
                    self.parameters.nu,
                    pv_set_point_real,
                    pv_set_point_imag,
                )
                # Updating Vk
                Ppv, Qpv = np.zeros(self.num_node), np.zeros(self.num_node)
                for pp, pv_ii in enumerate(self.pv_index):
                    Ppv[pv_ii] = Pk_last[pp]
                    Qpv[pv_ii] = Qk_last[pp]
                Ppv, Qpv = (
                    np.delete(Ppv, self.slack_bus),
                    np.delete(Qpv, self.slack_bus),
                )
                V_k = Vk_wopv + self.G @ Ppv + self.H @ Qpv
                # Updating duals
                dual_update_vv, lambdak_cur, muk_cur = primal_dual(
                    dual_update_v,
                    muk_last,
                    lambdak_last,
                    self.parameters.epsilon,
                    self.parameters.alpha,
                    uni_Vmax,
                    uni_Vmin,
                    V_k,
                )

                muk_last, lambdak_last = muk_cur, lambdak_cur
                dual_update_v = dual_update_vv
                # logger.debug(f"At iteration {jj}, the minimum vol is {np.min(V_k)}, the maximum vol is {np.max(V_k)}")
            # Do the projection
            Ppv, Qpv = np.zeros(self.num_node), np.zeros(self.num_node)
            for pp, pv_ii in enumerate(self.pv_index):
                pv_avail_v = self.pv_frame.iloc[pp]["avai"] / self.base_power
                pv_capacity_v = self.pv_frame.iloc[pp]["kVarRated"] / self.base_power
                Pk_last[pp], Qk_last[pp] = Proj_inverter(
                    Pk_last[pp], Qk_last[pp], pv_avail_v, pv_capacity_v
                )
                Ppv[pv_ii], Qpv[pv_ii] = Pk_last[pp], Qk_last[pp]
            Ppv, Qpv = np.delete(Ppv, self.slack_bus), np.delete(Qpv, self.slack_bus)
            V_hat_final = Vk_wopv + self.G @ Ppv + self.H @ Qpv

            # logger.debug(f"Before OMOO, the violated ones are {V_hat[ind]}")
            # logger.debug(f"After opf, approxiamted violations: {V_hat_final[ind]}")
            logger.debug(
                f"Target bounds are [{self.parameters.Vmax}, {self.parameters.Vmin}]"
            )
            return (
                Pk_last * self.base_power,
                Qk_last * self.base_power,
                True,
                V_hat_final,
            )


class OMOOFederate:
    "OMOO federate. Wraps optimal_pf with pubs and subs"

    def __init__(self, federate_name, algorithm_parameters, input_mapping):
        "Initializes federate with name and remaps input into subscriptions"
        deltat = 0.1

        self.algorithm_parameters = algorithm_parameters

        # Create Federate Info object that describes the federate properties #
        fedinfo = h.helicsCreateFederateInfo()

        fedinfo.core_name = federate_name
        fedinfo.core_type = h.HELICS_CORE_TYPE_ZMQ
        fedinfo.core_init = "--federates=1"
        h.helicsFederateInfoSetTimeProperty(
            fedinfo, h.helics_property_time_delta, deltat
        )

        self.vfed = h.helicsCreateValueFederate(federate_name, fedinfo)
        logger.info("Value federate created")

        # Register the publication #
        self.sub_voltages_real = self.vfed.register_subscription(
            input_mapping["voltages_real"], "V"
        )
        self.sub_voltages_imaginary = self.vfed.register_subscription(
            input_mapping["voltages_imag"], "V"
        )
        self.sub_power_P = self.vfed.register_subscription(
            input_mapping["powers_real"], "W"
        )
        self.sub_power_Q = self.vfed.register_subscription(
            input_mapping["powers_imag"], "W"
        )
        self.sub_topology = self.vfed.register_subscription(
            input_mapping["topology"], ""
        )
        self.injections = self.vfed.register_subscription(
            input_mapping["injections"], ""
        )
        self.sub_available_power = self.vfed.register_subscription(
            input_mapping["available_power"], ""
        )

        self.pub_voltage_mag = self.vfed.register_publication(
            "voltage_mag", h.HELICS_DATA_TYPE_STRING, ""
        )
        self.pub_voltage_angle = self.vfed.register_publication(
            "voltage_angle", h.HELICS_DATA_TYPE_STRING, ""
        )
        self.pub_P_set = self.vfed.register_publication(
            "P_set", h.HELICS_DATA_TYPE_STRING, ""
        )
        logger.debug("algorithm_parameters")
        logger.debug(algorithm_parameters)

    def run(self):
        "Enter execution and exchange data"
        # Enter execution mode #
        self.vfed.enter_executing_mode()
        logger.info("Entering execution mode")

        # granted_time = h.helicsFederateRequestTime(self.vfed, h.HELICS_TIME_MAXTIME)
        granted_time = h.helicsFederateRequestTime(self.vfed, 1000)

        topology = Topology.parse_obj(self.sub_topology.json)
        ids = topology.base_voltage_magnitudes.ids
        logger.info("Topology has been read")
        slack_index = None
        if not isinstance(topology.admittance, AdmittanceMatrix) and not isinstance(
            topology.admittance, AdmittanceSparse
        ):
            raise "OMOO algorithm expects AdmittanceMatrix/Sparse as input"

        for i in range(len(ids)):
            if ids[i] == topology.slack_bus[0]:
                slack_index = i

        if not slack_index == 0:
            raise "Slack index is not 0"

        slack_bus = [slack_index, slack_index + 1, slack_index + 2]
        Y = get_y(topology.admittance, topology.base_voltage_magnitudes.ids)
        if self.algorithm_parameters.units == UnitSystem.PER_UNIT:
            base_power = 100
            if self.algorithm_parameters.base_power != None:
                base_power = self.algorithm_parameters.base_power
            Y = (
                np.array(topology.base_voltage_magnitudes.values).reshape(1, -1)
                * Y
                * np.array(topology.base_voltage_magnitudes.values).reshape(-1, 1)
                / (base_power * 1000)
            )
        self.YLL = csc_matrix(
            np.delete(np.delete(Y, slack_bus, axis=0), slack_bus, axis=1)
        )
        self.YL0 = csc_matrix(np.delete(Y, slack_bus, axis=0)[:, slack_bus])
        self.Y = csc_matrix(Y)
        del Y

        ratings = eqarray_to_xarray(
            topology.injections.power_real
        ) + 1j * eqarray_to_xarray(topology.injections.power_imaginary)
        pv_ratings = ratings[ratings.equipment_ids.str.startswith("PVSystem")]

        v = measurement_to_xarray(topology.base_voltage_magnitudes)

        voltages = None
        power_P = None
        power_Q = None
        while granted_time < h.HELICS_TIME_MAXTIME:
            logger.debug("granted_time")
            logger.debug(granted_time)
            if not self.sub_voltages_real.is_updated():
                granted_time = h.helicsFederateRequestTime(
                    self.vfed, h.HELICS_TIME_MAXTIME
                )
                continue

            voltages_real = VoltagesReal.parse_obj(self.sub_voltages_real.json)
            voltages_imag = VoltagesImaginary.parse_obj(
                self.sub_voltages_imaginary.json
            )
            voltages = measurement_to_xarray(
                voltages_real
            ) + 1j * measurement_to_xarray(voltages_imag)
            logger.debug(np.max(np.abs(voltages) / v))
            assert topology.base_voltage_magnitudes.ids == list(voltages.ids.data)

            injections = Injection.parse_obj(self.injections.json)
            power_injections = eqarray_to_xarray(
                injections.power_real
            ) + 1j * eqarray_to_xarray(injections.power_imaginary)
            pv_injections = power_injections[
                power_injections.equipment_ids.str.startswith("PVSystem")
            ]
            _, pv_injections = xr.align(pv_ratings, pv_injections)
            available_power = measurement_to_xarray(
                MeasurementArray.parse_obj(self.sub_available_power.json)
            )

            split_power = available_power / pv_injections.ids.groupby(
                "equipment_ids"
            ).count().rename({"equipment_ids": "ids"})
            available_power = split_power.loc[
                pv_injections.equipment_ids
            ].assign_coords(ids=pv_injections.ids)

            pv = pd.DataFrame()
            pv["name"] = pv_ratings.equipment_ids.data
            pv["bus"] = pv_ratings.ids.data
            pv["kVarRated"] = pv_ratings.values.real

            # This needs to be fixed.
            pv["avai"] = available_power
            bus_to_index = {
                v: i for i, v in enumerate(topology.base_voltage_magnitudes.ids)
            }
            pv["index"] = [bus_to_index[v] for v in pv_injections.ids.data]

            V0 = voltages[slack_bus].data
            self.V0 = (
                V0 / np.array(topology.base_voltage_magnitudes.values)[slack_bus]
            ).reshape(3, -1)
            self.G, self.H, self.w_mag = getLinearModel(self.YLL, self.YL0, self.V0)

            logger.debug("PVframe")
            logger.debug(pv)

            power_P = PowersReal.parse_obj(self.sub_power_P.json)
            power_Q = PowersImaginary.parse_obj(self.sub_power_Q.json)
            assert topology.base_voltage_magnitudes.ids == power_P.ids
            assert topology.base_voltage_magnitudes.ids == power_Q.ids
            ts = time.time()
            opf = OMOO(
                self.algorithm_parameters,
                topology,
                slack_bus,
                V0,
                pv,
                self.YLL,
                self.YL0,
                self.G,
                self.H,
                self.w_mag,
            )

            P_set, Q_set, set_power, V_hat = opf.opf_run(
                np.abs(voltages), power_P, power_Q
            )
            power_set = P_set + 1j * Q_set
            power_factor = power_set.real / (np.abs(power_set) + 1e-7)
            pmpp = power_set.real / pv["kVarRated"]
            assert np.all(
                np.abs(power_factor) <= 1
            ), f"Invalid power factor at index {np.argmax(np.abs(power_factor) > 1)}: {power_factor[np.argmax(np.abs(power_factor) > 1)]}"
            assert np.all(
                (pmpp <= 1) & (pmpp >= 0)
            ), f"Invalid pmpp at index {np.argmax((pmpp > 1) | (pmpp < 0))}: {pmpp[np.argmax((pmpp > 1) | (pmpp < 0))]}"

            te = time.time()
            logger.debug(f"OMOO takes {(te-ts)/60} (min)")

            power_set_xr = (
                xr.DataArray(power_set, coords={"equipment_ids": pv.loc[:, "name"]})
                .groupby("equipment_ids")
                .sum()
            )

            available_total_xr = available_power.groupby("equipment_ids").sum()

            pv_settings = []
            command_list = []
            # We should test against the new interface
            for i in range(len(power_set_xr)):
                assert (
                    available_total_xr.equipment_ids[i] == power_set_xr.equipment_ids[i]
                )
                if np.isclose(available_total_xr[i], 0):
                    continue
                pv_settings.append(
                    (
                        power_set_xr.equipment_ids.data[i],
                        power_set_xr.values[i].real,
                        power_set_xr.values[i].imag,
                    )
                )
            command_list_obj = CommandList(__root__=command_list)
            logger.debug(command_list_obj)
            # Turn P_set and Q_set into commands
            if set_power:
                self.pub_P_set.publish(json.dumps(pv_settings))

            logger.info("end time: " + str(datetime.now()))

            # There should be a HELICS way to do this? Set resolution?
            previous_time = granted_time
            while (
                granted_time <= np.floor(previous_time) + 1
            ):  # This should avoid waiting a full 15 minutes
                granted_time = h.helicsFederateRequestTime(self.vfed, 1000)

        self.destroy()

    def destroy(self):
        "Finalize and destroy the federates"
        h.helicsFederateDisconnect(self.vfed)
        logger.info("Federate disconnected")

        h.helicsFederateFree(self.vfed)
        h.helicsCloseLibrary()


if __name__ == "__main__":
    with open("static_inputs.json") as f:
        config = json.load(f)
        federate_name = config["name"]
        if "algorithm_parameters" in config:
            parameters = OMOOParameters.parse_obj(config["algorithm_parameters"])
        else:
            parameters = OMOOParameters.parse_obj({})

    with open("input_mapping.json") as f:
        input_mapping = json.load(f)

    sfed = OMOOFederate(federate_name, parameters, input_mapping)
    sfed.run()
