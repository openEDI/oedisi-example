"""
Basic State Estimation Federate

Uses weighted least squares to estimate the voltage angles.

First `call_h` calculates the residual from the voltage magnitude and angle,
and `call_H` calculates a jacobian. Then `scipy.optimize.least_squares`
is used to solve.
"""
import helics as h
import json
import numpy as np
from pydantic import BaseModel
from typing import List
from scipy.optimize import least_squares

def cal_h(knownP, knownQ, knownV, Y, deltaK, VabsK, num_node):
    h1 = (VabsK[knownV]).reshape(-1,1)
    Vp = VabsK * np.exp(1j * deltaK)
    S = Vp * (Y.conjugate() @ Vp.conjugate())
    P, Q = S.real, S.imag
    h2, h3 = P[knownP].reshape(-1,1), Q[knownQ].reshape(-1,1)
    h = np.concatenate((h1, h2, h3), axis=0)
    return h.reshape(-1)

def cal_H(X0, z, num_node, knownP, knownQ, knownV, Y):
    deltaK, VabsK = X0[:num_node], X0[num_node:]
    num_knownV = len(knownV)
    #Calculate original H1
    H11, H12 = np.zeros((num_knownV, num_node)), np.zeros(num_knownV * num_node)
    H12[np.arange(num_knownV)*num_node + knownV] = 1
    H1 = np.concatenate((H11, H12.reshape(num_knownV, num_node)), axis=1)
    Vp = VabsK * np.exp(1j * deltaK)
##### S = np.diag(Vp) @ Y.conjugate() @ Vp.conjugate()
######  Take gradient with respect to V
    H_pow2 = (
            Vp.reshape(-1, 1) * Y.conjugate() * np.exp(-1j * deltaK).reshape(1, -1) + 
        np.exp(1j * deltaK) * np.diag(Y.conjugate() @ Vp.conjugate())
        )
    # Take gradient with respect to delta
    H_pow1 = (
            1j * Vp.reshape(-1, 1) * (np.diag(Y.conjugate() @ Vp.conjugate()) -
                Y.conjugate() * Vp.conjugate().reshape(1, -1))
            )
        
    H2 = np.concatenate((H_pow1.real, H_pow2.real), axis=1)[knownP, :]
    H3 = np.concatenate((H_pow1.imag, H_pow2.imag), axis=1)[knownQ, :]
    H = np.concatenate((H1, H2, H3), axis=0)   
    return -H

def residual(X0, z, num_node, knownP, knownQ, knownV, Y):
    delta, Vabs = X0[:num_node], X0[num_node:]
    h = cal_h(knownP, knownQ, knownV, Y, delta, Vabs, num_node)
    return z-h

class Complex(BaseModel):
    "Pydantic model for complex values with json representation"
    real: float
    imag: float


class Topology(BaseModel):
    "All necessary data for state estimator run"
    y_matrix: List[List[Complex]]
    phases: List[float]
    unique_ids: List[str]


class LabelledArray(BaseModel):
    "Labelled array has associated list of ids"
    array: List[float]
    unique_ids: List[str]


class PolarLabelledArray(BaseModel):
    "Labelled arrays of magnitudes and angles with list of ids"
    magnitudes: List[float]
    angles: List[float]
    unique_ids: List[str]


def matrix_to_numpy(y_matrix: List[List[Complex]]):
    "Convert list of list of our Complex type into a numpy matrix"
    return np.array([[x.real + 1j * x.imag for x in row] for row in y_matrix])


def get_indices(topology, labelled_array):
    "Get list of indices in the topology for each index of the labelled array"
    inv_map = {v: i for i, v in enumerate(topology.unique_ids)}
    return [inv_map[v] for v in labelled_array.unique_ids]


def state_estimator(topology, P, Q, V):
    """Estimates voltage magnitude and angle from topology, partial power injections
    P + Q i, and lossy partial voltage magnitude.

    Parameters
    ----------
    topology : Topology
        topology includes: Y-matrix, some initial phases, and unique ids
    P : LabelledArray
        Real power injection with unique ids
    Q : LabelledArray
        Reactive power injection with unique ids
    V : LabelledArray
        Voltage magnitude with unique ids
    """
    phase_res = np.array(topology.phases)
    num_node = len(topology.unique_ids)
    print(num_node)
    knownP = get_indices(topology, P)
    knownQ = get_indices(topology, Q)
    knownV = get_indices(topology, V)
    z = np.concatenate((V.array, P.array, Q.array), axis=0)

    delta, Vabs = np.zeros(num_node), np.ones(num_node)  # initialize targets
    X0 = np.concatenate((delta, Vabs))
    tol = 1e-5
    # Weights are ignored since errors are sampled from Gaussian
    res_1 = least_squares(
        residual,
        X0,
        jac=cal_H,
        verbose=2,
        ftol=tol,
        xtol=tol,
        gtol=tol,
        args=(z, num_node, knownP, knownQ, knownV, matrix_to_numpy(topology.y_matrix)),
    )
    result = res_1.x
    vmagestDecen, vangestDecen = result[num_node:], result[:num_node]
    vangestDecen = vangestDecen - vangestDecen[0] + phase_res
    return vmagestDecen, vangestDecen


class StateEstimatorFederate:
    "State estimator federate. Wraps state_estimation with pubs and subs"
    def __init__(self, federate_name, input_mapping):
        "Initializes federate with name and remaps input into subscriptions"
        deltat = 0.01

        # Create Federate Info object that describes the federate properties #
        fedinfo = h.helicsCreateFederateInfo()

        fedinfo.core_name = federate_name
        fedinfo.core_type = h.HELICS_CORE_TYPE_ZMQ
        fedinfo.core_init = "--federates=1"
        h.helicsFederateInfoSetTimeProperty(
            fedinfo, h.helics_property_time_delta, deltat
        )

        self.vfed = h.helicsCreateValueFederate(federate_name, fedinfo)
        print("Value federate created")

        # Register the publication #
        self.sub_voltages = self.vfed.register_subscription(
            input_mapping["voltages"], "V"
        )
        self.sub_power_P = self.vfed.register_subscription(
            input_mapping["power_P"], "W"
        )
        self.sub_power_Q = self.vfed.register_subscription(
            input_mapping["power_Q"], "W"
        )
        self.sub_topology = self.vfed.register_subscription(
            input_mapping["topology"], ""
        )
        self.pub_voltages = self.vfed.register_publication(
            "voltages", h.HELICS_DATA_TYPE_STRING, ""
        )

    def run(self):
        "Enter execution and exchange data"
        # Enter execution mode #
        self.vfed.enter_executing_mode()
        print("Entering execution mode")

        granted_time = -1

        for request_time in range(1, 100, 1):
            request_time += 0.5
            while granted_time < request_time:
                granted_time = h.helicsFederateRequestTime(self.vfed, request_time)

            print("Topology")
            topology = Topology.parse_obj(self.sub_topology.json)

            print("Voltages")
            voltages = LabelledArray.parse_obj(self.sub_voltages.json)

            print("P and Q")
            power_P = LabelledArray.parse_obj(self.sub_power_P.json)
            power_Q = LabelledArray.parse_obj(self.sub_power_Q.json)

            voltage_magnitudes, voltage_angles = state_estimator(
                topology, power_P, power_Q, voltages
            )
            voltage_results = PolarLabelledArray(
                magnitudes=list(voltage_magnitudes),
                angles=list(voltage_angles),
                unique_ids=topology.unique_ids,
            )

            self.pub_voltages.publish(voltage_results.json())

        self.destroy()

    def destroy(self):
        "Finalize and destroy the federates"
        h.helicsFederateDisconnect(self.vfed)
        print("Federate disconnected")

        h.helicsFederateFree(self.vfed)
        h.helicsCloseLibrary()


if __name__ == "__main__":
    with open("static_inputs.json") as f:
        config = json.load(f)
        federate_name = config["name"]

    with open("input_mapping.json") as f:
        input_mapping = json.load(f)

    sfed = StateEstimatorFederate(federate_name, input_mapping)
    sfed.run()
