from enum import Enum
import numpy as np
import cvxpy as cp
import math
import logging

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.DEBUG)

"""
Created on Tue March 30 11:11:21 2021

@author:  Rabayet
"""


"""

Updated:

1. Dist_flag is default False
2. Primary KV level variable is introduced
3. DER maximization without alpha parameter.

4. Hard coded for the split-phase using last char. a,b or c from OpenDSS

5. Q control added
6. Changed BaseS to 1MW and Base kV

"""


class ControlType(Enum):
    WATT = 1
    VAR = 2
    WATT_VAR = 3


def ignore_phase(control: dict) -> float:
    setpoint = 0
    for key, val in control.items():
        if np.abs(val) > np.abs(setpoint):
            setpoint = val
    return setpoint


def power_balance(A, b, k_frm, k_to, counteq, col, val):
    for k in k_frm:
        A[counteq, col + k] = -1
    for k in k_to:
        A[counteq, col + k] = 1

    A[counteq, val] = -1
    b[counteq] = 0
    return A, b


def voltage_cons_pri(
    A,
    b,
    p,
    frm,
    to,
    counteq,
    pii,
    qii,
    pij,
    qij,
    pik,
    qik,
    nbus_ABC,
    nbus_s1s2,
    nbranch_ABC,
    baseZ,
):
    A[counteq, frm] = 1
    A[counteq, to] = -1
    n_flow_ABC = (nbus_ABC * 3 + nbus_s1s2) + (nbus_ABC * 6 + nbus_s1s2 * 2)
    # real power drop
    A[counteq, p + n_flow_ABC + +nbranch_ABC * 0] = pii / baseZ
    A[counteq, p + n_flow_ABC + nbranch_ABC * 1] = pij / baseZ
    A[counteq, p + n_flow_ABC + nbranch_ABC * 2] = pik / baseZ
    # reactive power drop
    A[counteq, p + n_flow_ABC + nbranch_ABC * 3] = qii / baseZ
    A[counteq, p + n_flow_ABC + nbranch_ABC * 4] = qij / baseZ
    A[counteq, p + n_flow_ABC + nbranch_ABC * 5] = qik / baseZ
    b[counteq] = 0.0
    return A, b


def optimal_power_flow(
    branch_info: dict,
    bus_info: dict,
    source_bus: str,
    control: ControlType,
    pf_flag: bool,
):
    # System's base definition
    BASE_S = 1 / (1000000 * 100)
    S_CAPACITY = 1.2
    PRIMARY_V = 0.12
    SOURCE_V = [1.04, 1.04, 1.04]
    basekV = bus_info[source_bus]["kv"] / np.sqrt(3)
    baseZ = basekV**2 / 100

    # Find the ABC phase and s1s2 phase triplex line and bus numbers
    nbranch_ABC = 0
    nbus_ABC = 0
    nbranch_s1s2 = 0
    nbus_s1s2 = 0
    mult = 1
    secondary_model = ["TPX_LINE", "SPLIT_PHASE"]
    name = []
    for b_eq in branch_info:
        if branch_info[b_eq]["type"] in secondary_model:
            nbranch_s1s2 += 1
        else:
            nbranch_ABC += 1

    for b_eq in bus_info:
        name.append(b_eq)
        if bus_info[b_eq]["kv"] > PRIMARY_V:
            nbus_ABC += 1
        else:
            nbus_s1s2 += 1

    # Number of Optimization Variables
    voltage_count = (nbus_ABC * 3 + nbus_s1s2) + (nbus_ABC * 6 + nbus_s1s2 * 2)
    injection_count = nbranch_ABC * 6 + nbranch_s1s2 * 2
    flow_count = nbus_ABC * 3 + nbus_s1s2
    der_count = (nbus_ABC * 3 + nbus_s1s2) + nbus_ABC * 3 + nbus_s1s2
    variable_number = voltage_count + injection_count + flow_count + der_count

    # Number of equality/inequality constraints (Injection equations (ABC) at each bus)
    #    #  Check if this is correct number or not:
    n_bus = nbus_ABC * 3 + nbus_s1s2  # Total Bus Number
    n_branch = nbranch_ABC * 3 + nbranch_s1s2  # Total Branch Number

    constraint_number = 1000 + variable_number + n_bus + 3 * n_bus + n_branch
    A_ineq = np.zeros((constraint_number, variable_number))
    b_ineq = np.zeros(constraint_number)

    x = cp.Variable(variable_number)
    # Initialize the matrices
    P = np.zeros((variable_number, variable_number))
    q_obj_vector = np.zeros(variable_number)
    A_eq = np.zeros((constraint_number, variable_number))
    b_eq = np.zeros(constraint_number)

    # Some extra variable definition for clean code:
    #                      Voltage      PQ_inj     PQ_flow
    state_variable_number = n_bus + 2 * n_bus + 2 * n_branch

    # Q-dg variable starting number
    #                               P_dg
    n_Qdg = state_variable_number + n_bus
    # s = 0

    # # Deciding the starting index for Control Variable
    # if P_control==True:
    #     variable_start_idx = state_variable_number
    # elif Q_control==True:
    #     variable_start_idx = n_Qdg

    # Linear Programming Cost Vector:
    # for k in range(nbus_ABC  + nbus_s1s2):

    if control is ControlType.WATT:
        for k in range(n_bus):
            q_obj_vector[state_variable_number + k] = -1  # DER max objective
    elif control is ControlType.VAR:
        for k in range(n_bus):
            q_obj_vector[n_Qdg + k] = 0  # Just Voltage regulation
            # q_obj_vector[n_Qdg + k] = -1  # Just Voltage regulation

    # # Define BFM constraints for both real and reactive power: Power flow conservaion
    # Constraint 1: Flow equation

    # sum(Sij) - sum(Sjk) == -sj

    counteq = 0
    for keyb, val_bus in bus_info.items():
        if keyb != source_bus:
            k_frm_3p = []
            k_to_3p = []
            k_frm_1p = []
            k_frm_1pa, k_frm_1pb, k_frm_1pc = [], [], []
            k_frm_1qa, k_frm_1qb, k_frm_1qc = [], [], []
            k_to_1p = []

            # Find bus idx in "from" of branch_sw_data
            ind_frm = 0
            ind_to = 0
            if val_bus["kv"] < PRIMARY_V:
                for key, val_br in branch_info.items():
                    if val_bus["idx"] == val_br["from"]:
                        k_frm_1p.append(ind_frm - nbranch_ABC)

                    if val_bus["idx"] == val_br["to"]:
                        k_to_1p.append(ind_to - nbranch_ABC)
                    ind_to += 1
                    ind_frm += 1
                loc = (
                    (nbus_ABC * 3 + nbus_s1s2)
                    + (nbus_ABC * 6 + nbus_s1s2 * 2)
                    + nbranch_ABC * 6
                )
                A_eq, b_eq = power_balance(
                    A_eq,
                    b_eq,
                    k_frm_1p,
                    k_to_1p,
                    counteq,
                    loc,
                    val_bus["idx"] + nbus_ABC * 3 + nbus_s1s2 + nbus_ABC * 5,
                )
                counteq += 1
                A_eq, b_eq = power_balance(
                    A_eq,
                    b_eq,
                    k_frm_1p,
                    k_to_1p,
                    counteq,
                    loc + nbranch_s1s2,
                    val_bus["idx"]
                    + nbus_ABC * 3
                    + nbus_s1s2
                    + nbus_ABC * 5
                    + nbus_s1s2,
                )
                counteq += 1
            else:
                for key, val_br in branch_info.items():
                    if val_bus["idx"] == val_br["from"]:
                        if bus_info[val_br["to_bus"]]["kv"] > PRIMARY_V:
                            k_frm_3p.append(ind_frm)
                        else:
                            if key[-1] == "a":
                                k_frm_1pa.append(
                                    nbranch_ABC * 6 + ind_frm - nbranch_ABC
                                )
                                k_frm_1qa.append(
                                    nbranch_ABC * 3
                                    + ind_frm
                                    - nbranch_ABC
                                    + nbranch_s1s2
                                )
                            if key[-1] == "b":
                                k_frm_1pb.append(
                                    nbranch_ABC * 5 + ind_frm - nbranch_ABC
                                )
                                k_frm_1qb.append(
                                    nbranch_ABC * 2
                                    + ind_frm
                                    - nbranch_ABC
                                    + nbranch_s1s2
                                )
                            if key[-1] == "c":
                                k_frm_1pc.append(
                                    nbranch_ABC * 4 + ind_frm - nbranch_ABC
                                )
                                k_frm_1qc.append(
                                    nbranch_ABC * 1
                                    + ind_frm
                                    - nbranch_ABC
                                    + nbranch_s1s2
                                )

                    if val_bus["idx"] == val_br["to"]:
                        if bus_info[val_br["fr_bus"]]["kv"] > PRIMARY_V:
                            k_to_3p.append(ind_to)
                        else:
                            k_to_1p.append(ind_to - nbranch_ABC)
                    ind_to += 1
                    ind_frm += 1
                loc = (nbus_ABC * 3 + nbus_s1s2) + (nbus_ABC * 6 + nbus_s1s2 * 2)
                # Finding the kfrms
                k_frm_A = k_frm_3p + k_frm_1pa
                k_frm_B = k_frm_3p + k_frm_1pb
                k_frm_C = k_frm_3p + k_frm_1pc
                k_to_A = k_to_B = k_to_C = k_to_3p

                # Real Power balance equations
                # # Phase A
                A_eq, b_eq = power_balance(
                    A_eq,
                    b_eq,
                    k_frm_A,
                    k_to_A,
                    counteq,
                    loc,
                    val_bus["idx"] + nbus_ABC * 3 + nbus_s1s2 + nbus_ABC * 0,
                )
                counteq += 1
                # # # Phase B
                A_eq, b_eq = power_balance(
                    A_eq,
                    b_eq,
                    k_frm_B,
                    k_to_B,
                    counteq,
                    loc + nbranch_ABC,
                    val_bus["idx"] + nbus_ABC * 3 + nbus_s1s2 + nbus_ABC * 1,
                )
                counteq += 1
                # # # Phase C
                A_eq, b_eq = power_balance(
                    A_eq,
                    b_eq,
                    k_frm_C,
                    k_to_C,
                    counteq,
                    loc + nbranch_ABC * 2,
                    val_bus["idx"] + nbus_ABC * 3 + nbus_s1s2 + nbus_ABC * 2,
                )
                counteq += 1
                k_frm_A = k_frm_3p + k_frm_1qa
                k_frm_B = k_frm_3p + k_frm_1qb
                k_frm_C = k_frm_3p + k_frm_1qc

                # Reactive Power balance equations
                loc = (
                    (nbus_ABC * 3 + nbus_s1s2)
                    + (nbus_ABC * 6 + nbus_s1s2 * 2)
                    + nbranch_ABC * 3
                )
                # Phase A
                A_eq, b_eq = power_balance(
                    A_eq,
                    b_eq,
                    k_frm_A,
                    k_to_A,
                    counteq,
                    loc,
                    val_bus["idx"] + nbus_ABC * 3 + nbus_s1s2 + nbus_ABC * 3,
                )
                counteq += 1
                # Phase B
                A_eq, b_eq = power_balance(
                    A_eq,
                    b_eq,
                    k_frm_B,
                    k_to_B,
                    counteq,
                    loc + nbranch_ABC,
                    val_bus["idx"] + nbus_ABC * 3 + nbus_s1s2 + nbus_ABC * 4,
                )
                counteq += 1
                # Phase C
                A_eq, b_eq = power_balance(
                    A_eq,
                    b_eq,
                    k_frm_C,
                    k_to_C,
                    counteq,
                    loc + nbranch_ABC * 2,
                    val_bus["idx"] + nbus_ABC * 3 + nbus_s1s2 + nbus_ABC * 5,
                )
                counteq += 1

    # Constraint 2: Voltage drop equation:
    # Vj = Vi - Zij Sij* - Sij Zij*

    # For Primary Nodes:
    idx = 0
    v_lim = []
    for k, val_br in branch_info.items():
        # Not writing voltage constraints for transformers
        if val_br["type"] not in secondary_model:
            z = np.asarray(val_br["zprim"])
            v_lim.append(val_br["from"])
            v_lim.append(val_br["to"])
            # Writing three phase voltage constraints
            # Phase A
            paa, qaa = -2 * z[0, 0][0], -2 * z[0, 0][1]
            pab, qab = (
                -(-z[0, 1][0] + math.sqrt(3) * z[0, 1][1]),
                -(-z[0, 1][1] - math.sqrt(3) * z[0, 1][0]),
            )
            pac, qac = (
                -(-z[0, 2][0] - math.sqrt(3) * z[0, 2][1]),
                -(-z[0, 2][1] + math.sqrt(3) * z[0, 2][0]),
            )
            A_eq, b_eq = voltage_cons_pri(
                A_eq,
                b_eq,
                idx,
                val_br["from"],
                val_br["to"],
                counteq,
                paa,
                qaa,
                pab,
                qab,
                pac,
                qac,
                nbus_ABC,
                nbus_s1s2,
                nbranch_ABC,
                baseZ,
            )

            counteq += 1
            # Phase B
            pbb, qbb = -2 * z[1, 1][0], -2 * z[1, 1][1]
            pba, qba = (
                -(-z[0, 1][0] - math.sqrt(3) * z[0, 1][1]),
                -(-z[0, 1][1] + math.sqrt(3) * z[0, 1][0]),
            )
            pbc, qbc = (
                -(-z[1, 2][0] + math.sqrt(3) * z[1, 2][1]),
                -(-z[1, 2][1] - math.sqrt(3) * z[1, 2][0]),
            )
            A_eq, b_eq = voltage_cons_pri(
                A_eq,
                b_eq,
                idx,
                nbus_ABC + val_br["from"],
                nbus_ABC + val_br["to"],
                counteq,
                pba,
                qba,
                pbb,
                qbb,
                pbc,
                qbc,
                nbus_ABC,
                nbus_s1s2,
                nbranch_ABC,
                baseZ,
            )
            counteq += 1
            # Phase C
            pcc, qcc = -2 * z[2, 2][0], -2 * z[2, 2][1]
            pca, qca = (
                -(-z[0, 2][0] + math.sqrt(3) * z[0, 2][1]),
                -(-z[0, 2][1] - math.sqrt(3) * z[0, 2][0]),
            )
            pcb, qcb = (
                -(-z[1, 2][0] - math.sqrt(3) * z[1, 2][1]),
                -(-z[0, 2][1] + math.sqrt(3) * z[1, 2][0]),
            )
            A_eq, b_eq = voltage_cons_pri(
                A_eq,
                b_eq,
                idx,
                nbus_ABC * 2 + val_br["from"],
                nbus_ABC * 2 + val_br["to"],
                counteq,
                pca,
                qca,
                pcb,
                qcb,
                pcc,
                qcc,
                nbus_ABC,
                nbus_s1s2,
                nbranch_ABC,
                baseZ,
            )
            counteq += 1
        idx += 1

    # For secondary Nodes:
    def voltage_cons_sec(A, b, p, frm, to, counteq, p_pri, q_pri, p_sec, q_sec):
        A[counteq, frm] = 1
        A[counteq, to] = -1
        n_flow_s1s2 = (
            (nbus_ABC * 3 + nbus_s1s2)
            + (nbus_ABC * 6 + nbus_s1s2 * 2)
            + nbranch_ABC * 6
        )
        # real power drop
        A[counteq, p + n_flow_s1s2] = p_pri + 0.5 * p_sec
        # reactive power drop
        A[counteq, p + n_flow_s1s2 + nbranch_s1s2] = q_pri + 0.5 * q_sec
        b[counteq] = 0.0
        return A, b

    idx = 0
    pq_index = []
    for k, val_br in branch_info.items():
        # For split phase transformer, we use interlace design
        if val_br["type"] in secondary_model:
            if val_br["type"] == "SPLIT_PHASE":
                pq_index.append(val_br["idx"])
                zp = np.asarray(val_br["impedance"])
                zs = np.asarray(val_br["impedance1"])
                v_lim.append(val_br["from"])
                v_lim.append(val_br["to"])
                # Writing voltage constraints
                # Phase S1
                p_pri, q_pri = -2 * zp[0], -2 * zp[1]
                p_sec, q_sec = -2 * zs[0], -2 * zs[1]
                phase = k[-1]
                if phase == "a":
                    from_bus = val_br["from"]
                if phase == "b":
                    from_bus = val_br["from"] + nbus_ABC
                if phase == "c":
                    from_bus = val_br["from"] + nbus_ABC * 2
                to_bus = val_br["to"] - nbus_ABC + nbus_ABC * 3
                # A, b = voltage_cons(A, b, idx - nbus_ABC, from_bus, to_bus, counteq, p_pri, q_pri, p_sec, q_sec)
                A_eq, b_eq = voltage_cons_sec(
                    A_eq,
                    b_eq,
                    idx - nbranch_ABC,
                    from_bus,
                    to_bus,
                    counteq,
                    p_pri,
                    q_pri,
                    p_sec,
                    q_sec,
                    nbus_ABC,
                    nbus_s1s2,
                    nbranch_ABC,
                    baseZ,
                )  # Monish
                counteq += 1

            # For triplex line, we assume there is no mutual coupling
            if val_br["type"] != "SPLIT_PHASE":
                # The impedance of line will be converted into pu here.
                zbase = 120.0 * 120.0 / 15000
                zp = np.asarray(val_br["impedance"])
                v_lim.append(val_br["from"])
                v_lim.append(val_br["to"])
                # Writing voltage constraints
                # Phase S1
                p_s1, q_s1 = 0, 0
                p_s2, q_s2 = -2 * zp[0, 0][0] / zbase, -2 * zp[0, 0][1] / zbase
                from_bus = val_br["from"] - nbus_ABC + nbus_ABC * 3
                to_bus = val_br["to"] - nbus_ABC + nbus_ABC * 3
                # A, b = voltage_cons(A, b, idx - nbus_ABC, from_bus, to_bus, counteq, p_s1, q_s1, p_s2, q_s2)
                A_eq, b_eq = voltage_cons_sec(
                    A_eq,
                    b_eq,
                    idx - nbranch_ABC,
                    from_bus,
                    to_bus,
                    counteq,
                    p_s1,
                    q_s1,
                    p_s2,
                    q_s2,
                    nbus_ABC,
                    nbus_s1s2,
                    nbranch_ABC,
                    baseZ,
                )  # Monish
                counteq += 1
        idx += 1

    # Constraint 3: Substation voltage definition
    # V_substation = V_source
    source_bus_idx = bus_info[source_bus]["idx"]
    A_eq[counteq, source_bus_idx] = 1
    b_eq[counteq] = (SOURCE_V[0]) ** 2
    counteq += 1
    A_eq[counteq, source_bus_idx + nbus_ABC] = 1
    b_eq[counteq] = (SOURCE_V[1]) ** 2
    counteq += 1
    A_eq[counteq, source_bus_idx + nbus_ABC * 2] = 1
    b_eq[counteq] = (SOURCE_V[2]) ** 2
    counteq += 1

    # BFM Power Flow Model ends. Next we define the Control variables:

    # # Make injection a decision variable

    # print("Start: Injection Constraints")

    # P_dg control:
    if control is ControlType.WATT:
        DG_up_lim = np.zeros((n_bus, 1))
        for keyb, val_bus in bus_info.items():
            if keyb != source_bus:
                # Real power injection at a bus
                if val_bus["kv"] > PRIMARY_V:
                    # p_inj  + p_gen(control var) =  p_load
                    # Phase A Real Power
                    A_eq[
                        counteq,
                        nbus_ABC * 3 + nbus_s1s2 + nbus_ABC * 0 + val_bus["idx"],
                    ] = 1
                    A_eq[
                        counteq, state_variable_number + nbus_ABC * 0 + val_bus["idx"]
                    ] = 1
                    b_eq[counteq] = val_bus["pq"][0][0] * BASE_S * mult
                    counteq += 1
                    # Phase B Real Power
                    A_eq[
                        counteq,
                        nbus_ABC * 3 + nbus_s1s2 + nbus_ABC * 1 + val_bus["idx"],
                    ] = 1
                    A_eq[
                        counteq, state_variable_number + nbus_ABC * 1 + val_bus["idx"]
                    ] = 1
                    b_eq[counteq] = val_bus["pq"][1][0] * BASE_S * mult
                    counteq += 1
                    # Phase C Real Power
                    A_eq[
                        counteq,
                        nbus_ABC * 3 + nbus_s1s2 + nbus_ABC * 2 + val_bus["idx"],
                    ] = 1
                    A_eq[
                        counteq, state_variable_number + nbus_ABC * 2 + val_bus["idx"]
                    ] = 1
                    b_eq[counteq] = val_bus["pq"][2][0] * BASE_S * mult
                    counteq += 1

                    # DG upper limit set up:
                    DG_up_lim[nbus_ABC * 0 + val_bus["idx"]] = (
                        val_bus["pv"][0][0] * BASE_S
                    )
                    DG_up_lim[nbus_ABC * 1 + val_bus["idx"]] = (
                        val_bus["pv"][1][0] * BASE_S
                    )
                    DG_up_lim[nbus_ABC * 2 + val_bus["idx"]] = (
                        val_bus["pv"][2][0] * BASE_S
                    )

                    # Phase A Reactive power
                    A_eq[
                        counteq,
                        nbus_ABC * 3 + nbus_s1s2 + nbus_ABC * 3 + val_bus["idx"],
                    ] = 1
                    b_eq[counteq] = val_bus["pq"][0][1] * BASE_S * mult
                    counteq += 1
                    # Phase B Reactive power
                    A_eq[
                        counteq,
                        nbus_ABC * 3 + nbus_s1s2 + nbus_ABC * 4 + val_bus["idx"],
                    ] = 1
                    b_eq[counteq] = val_bus["pq"][1][1] * BASE_S * mult
                    counteq += 1
                    # Phase C Reactive power
                    A_eq[
                        counteq,
                        nbus_ABC * 3 + nbus_s1s2 + nbus_ABC * 5 + val_bus["idx"],
                    ] = 1
                    b_eq[counteq] = val_bus["pq"][2][1] * BASE_S * mult
                    counteq += 1

                # work on this for the secondary netowrks:
                else:
                    A_eq[
                        counteq,
                        nbus_ABC * 3 + nbus_s1s2 + nbus_ABC * 5 + val_bus["idx"],
                    ] = 1
                    A_eq[counteq, state_variable_number + val_bus["idx"]] = 1
                    b_eq[counteq] = val_bus["pq"][0] * BASE_S * mult
                    counteq += 1
                    # Reactive power
                    A_eq[
                        counteq,
                        nbus_ABC * 3
                        + nbus_s1s2
                        + nbus_ABC * 5
                        + nbus_s1s2
                        + val_bus["idx"],
                    ] = 1
                    A_eq[counteq, n_Qdg + nbus_ABC * 2 + val_bus["idx"]] = -1
                    b_eq[counteq] = val_bus["pq"][1] * BASE_S
                    counteq += 1

                    DG_up_lim[nbus_ABC * 3 + val_bus["idx"]] = val_bus["pv"][0] * BASE_S

    elif control is ControlType.VAR:
        DG_up_lim = np.zeros((n_bus, 1))
        for keyb, val_bus in bus_info.items():
            if keyb != source_bus:
                # Real power injection at a bus
                if val_bus["kv"] > PRIMARY_V:
                    # p_inj   =  - p_d_gen + p_load
                    # Phase A Real Power
                    A_eq[
                        counteq,
                        nbus_ABC * 3 + nbus_s1s2 + nbus_ABC * 0 + val_bus["idx"],
                    ] = 1
                    b_eq[counteq] = (
                        -val_bus["pv"][0][0] * BASE_S
                        + val_bus["pq"][0][0] * BASE_S * mult
                    )
                    counteq += 1
                    # Phase B Real Power
                    A_eq[
                        counteq,
                        nbus_ABC * 3 + nbus_s1s2 + nbus_ABC * 1 + val_bus["idx"],
                    ] = 1
                    b_eq[counteq] = (
                        -val_bus["pv"][1][0] * BASE_S
                        + val_bus["pq"][1][0] * BASE_S * mult
                    )
                    counteq += 1
                    # Phase C Real Power
                    A_eq[
                        counteq,
                        nbus_ABC * 3 + nbus_s1s2 + nbus_ABC * 2 + val_bus["idx"],
                    ] = 1
                    b_eq[counteq] = (
                        -val_bus["pv"][2][0] * BASE_S
                        + val_bus["pq"][2][0] * BASE_S * mult
                    )
                    counteq += 1

                    # Q_inj  + Q_gen(control var) =  Q_load
                    # Phase A Reactive power
                    A_eq[
                        counteq,
                        nbus_ABC * 3 + nbus_s1s2 + nbus_ABC * 3 + val_bus["idx"],
                    ] = 1
                    A_eq[counteq, n_Qdg + nbus_ABC * 0 + val_bus["idx"]] = 1
                    b_eq[counteq] = val_bus["pq"][0][1] * BASE_S * mult
                    counteq += 1
                    # Phase B Reactive power
                    A_eq[
                        counteq,
                        nbus_ABC * 3 + nbus_s1s2 + nbus_ABC * 4 + val_bus["idx"],
                    ] = 1
                    A_eq[counteq, n_Qdg + nbus_ABC * 1 + val_bus["idx"]] = 1
                    b_eq[counteq] = val_bus["pq"][1][1] * BASE_S * mult
                    counteq += 1
                    # Phase C Reactive power
                    A_eq[
                        counteq,
                        nbus_ABC * 3 + nbus_s1s2 + nbus_ABC * 5 + val_bus["idx"],
                    ] = 1
                    A_eq[counteq, n_Qdg + nbus_ABC * 2 + val_bus["idx"]] = 1
                    b_eq[counteq] = val_bus["pq"][2][1] * BASE_S * mult
                    counteq += 1

                    # DG upper limit set up:
                    DG_up_lim[nbus_ABC * 0 + val_bus["idx"]] = np.sqrt(
                        ((S_CAPACITY * val_bus["pv"][0][0] * BASE_S) ** 2)
                        - ((val_bus["pv"][0][0] * BASE_S) ** 2)
                    )
                    DG_up_lim[nbus_ABC * 1 + val_bus["idx"]] = np.sqrt(
                        ((S_CAPACITY * val_bus["pv"][1][0] * BASE_S) ** 2)
                        - ((val_bus["pv"][1][0] * BASE_S) ** 2)
                    )
                    DG_up_lim[nbus_ABC * 2 + val_bus["idx"]] = np.sqrt(
                        ((S_CAPACITY * val_bus["pv"][2][0] * BASE_S) ** 2)
                        - ((val_bus["pv"][2][0] * BASE_S) ** 2)
                    )

                # work on this for the secondary netowrks:
                else:
                    A_eq[
                        counteq,
                        nbus_ABC * 3 + nbus_s1s2 + nbus_ABC * 5 + val_bus["idx"],
                    ] = 1
                    A_eq[counteq, state_variable_number + val_bus["idx"]] = 1
                    b_eq[counteq] = val_bus["pq"][0] * BASE_S * mult
                    counteq += 1
                    # Reactive power
                    A_eq[
                        counteq,
                        nbus_ABC * 3
                        + nbus_s1s2
                        + nbus_ABC * 5
                        + nbus_s1s2
                        + val_bus["idx"],
                    ] = 1
                    A_eq[counteq, n_Qdg + nbus_ABC * 2 + val_bus["idx"]] = -1
                    b_eq[counteq] = val_bus["pq"][1] * BASE_S
                    counteq += 1

                    DG_up_lim[nbus_ABC * 3 + val_bus["idx"]] = val_bus["pv"][0] * BASE_S

    # Reactive power as a function of real power and inverter rating
    countineq = 0

    for keyb, val_bus in bus_info.items():
        if val_bus["kv"] < PRIMARY_V:
            A_eq[counteq, n_Qdg + nbus_ABC * 2 + val_bus["idx"]] = 1
            b_eq[counteq] = 0.0 * val_bus["s_rated"] * BASE_S
            counteq += 1

    # Constraints for all bound within Maximum Capacity values
    # Only P_dg control Variable:
    if control is ControlType.WATT:
        for k in range(n_bus):
            A_ineq[countineq, state_variable_number + k] = 1
            b_ineq[countineq] = DG_up_lim[k, 0]
            countineq += 1

        for k in range(n_bus):
            A_ineq[countineq, state_variable_number + k] = -1
            b_ineq[countineq] = 0.0
            countineq += 1

    # Only Q_dg control Variable:
    elif control is ControlType.VAR:
        for k in range(n_bus):
            A_ineq[countineq, n_Qdg + k] = 1
            b_ineq[countineq] = DG_up_lim[k, 0]
            countineq += 1

        for k in range(n_bus):
            A_ineq[countineq, n_Qdg + k] = -1
            b_ineq[countineq] = DG_up_lim[k, 0]
            countineq += 1

    # Constraint 3: 0.95^2 <= V <= 1.05^2 (For those nodes where voltage constraint exist)
    # print("Formulating voltage limit constraints")
    v_idxs = list(set(v_lim))
    # # Does the vmin make sense here?
    if pf_flag is True:
        vmax = 1.5
        vmin = 0.1
    else:
        vmax = 1.05
        vmin = 0.95

    for k in range(nbus_ABC):
        if k in v_idxs:
            # Upper bound
            A_ineq[countineq, k] = 1
            b_ineq[countineq] = vmax**2
            countineq += 1
            A_ineq[countineq, k + nbus_ABC] = 1
            b_ineq[countineq] = vmax**2
            countineq += 1
            A_ineq[countineq, k + nbus_ABC * 2] = 1
            b_ineq[countineq] = vmax**2
            countineq += 1
            # Lower Bound
            A_ineq[countineq, k] = -1
            b_ineq[countineq] = -(vmin**2)
            countineq += 1
            A_ineq[countineq, k + nbus_ABC] = -1
            b_ineq[countineq] = -(vmin**2)
            countineq += 1
            A_ineq[countineq, k + nbus_ABC * 2] = -1
            b_ineq[countineq] = -(vmin**2)
            countineq += 1

    prob = cp.Problem(
        cp.Minimize(q_obj_vector.T @ x), [A_ineq @ x <= b_ineq, A_eq @ x == b_eq]
    )

    prob.solve(solver=cp.ECOS, verbose=True)
    logger.info(prob.status)

    if prob.status == "infeasible_or_unbounded" or prob.status == "infeasible":
        logger.debug("Check for limits. Power flow didn't converge")
        exit()

    from_bus = []
    to_bus = []
    name = []

    for key, val_br in branch_info.items():
        from_bus.append(val_br["fr_bus"])
        to_bus.append(val_br["to_bus"])
        name.append(key)

    i = 0
    mul = 1 / (BASE_S * 1000)
    line_flow = {}
    n_flow_ABC = (nbus_ABC * 3 + nbus_s1s2) + (nbus_ABC * 6 + nbus_s1s2 * 2)
    for k in range(n_flow_ABC, n_flow_ABC + nbranch_ABC):
        line_flow[name[i]] = {}
        line_flow[name[i]]["A"] = [
            x.value[k] * mul * 1000,
            x.value[k + nbranch_ABC * 3] * mul * 1000,
        ]
        line_flow[name[i]]["B"] = [
            x.value[k + nbranch_ABC] * mul * 1000,
            x.value[k + nbranch_ABC * 4] * mul * 1000,
        ]
        line_flow[name[i]]["C"] = [
            x.value[k + nbranch_ABC * 2] * mul * 1000,
            x.value[k + nbranch_ABC * 5] * mul * 1000,
        ]
        i += 1

    n_flow_s1s2 = (
        (nbus_ABC * 3 + nbus_s1s2) + (nbus_ABC * 6 + nbus_s1s2 * 2) + nbranch_ABC * 6
    )
    name = []
    for key, val_br in bus_info.items():
        name.append(key)
    bus_voltage = {}
    i = 0
    for k in range(nbus_ABC):
        bus_voltage[name[k]] = {}
        bus_voltage[name[k]]["A"] = math.sqrt(abs(x.value[k]))
        bus_voltage[name[k]]["B"] = math.sqrt(abs(x.value[nbus_ABC + k]))
        bus_voltage[name[k]]["C"] = math.sqrt(abs(x.value[nbus_ABC * 2 + k]))
        i += 1

    # Monish Edits
    for key, val_bus in bus_info.items():
        # volt.append(
        #     [name[k], '{:.4f}'.format(math.sqrt(abs(x.value[k]))),
        #      '{:.4f}'.format(math.sqrt(abs(x.value[nbus_ABC + k]))),
        #      '{:.4f}'.format(math.sqrt(abs(x.value[nbus_ABC * 2 + k])))])
        bus_voltage[key] = {}
        bus_voltage[key]["A"] = math.sqrt(abs(x.value[val_bus["idx"]]))
        bus_voltage[key]["B"] = math.sqrt(abs(x.value[nbus_ABC + val_bus["idx"]]))
        bus_voltage[key]["C"] = math.sqrt(abs(x.value[nbus_ABC * 2 + val_bus["idx"]]))
        i += 1

    if control is ControlType.WATT:
        control_variable_idx_start = state_variable_number
    elif control is ControlType.VAR:
        control_variable_idx_start = n_Qdg

    generation_output = np.zeros((nbus_ABC, 3))
    for k in range(nbus_ABC * 3):
        # injection.append([name[k], '{:.4f}'.format((x.value[k + state_variable_number]))])
        if DG_up_lim[k, 0]:
            if k < nbus_ABC:
                generation_output[k, 0] = x.value[k + control_variable_idx_start]
            elif k < (nbus_ABC * 2):
                generation_output[k - nbus_ABC, 1] = x.value[
                    k + control_variable_idx_start
                ]
            else:
                generation_output[k - (nbus_ABC * 2), 2] = x.value[
                    k + control_variable_idx_start
                ]
    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    opf_control_variable = {}
    for key, val_bus in bus_info.items():
        opf_control_variable[key] = {}
        opf_control_variable[key]["A"] = x.value[
            val_bus["idx"] + control_variable_idx_start
        ]
        opf_control_variable[key]["B"] = x.value[
            nbus_ABC + val_bus["idx"] + control_variable_idx_start
        ]
        opf_control_variable[key]["C"] = x.value[
            nbus_ABC * 2 + val_bus["idx"] + control_variable_idx_start
        ]

    kw_converter = 1 / BASE_S / 1000

    return bus_voltage, line_flow, opf_control_variable, kw_converter
