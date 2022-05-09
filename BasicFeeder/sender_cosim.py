import logging
import helics as h
import opendssdirect as dss
import sys
import pandas as pd
import json
from dss_functions import snapshot_run
from FeederSimulator import FeederSimulator, FeederConfig
from pydantic import BaseModel
from typing import List
import numpy as np

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)

test_se = False

class Complex(BaseModel):
    real: float
    imag: float


class Topology(BaseModel):
    y_matrix: List[List[Complex]]
    phases: List[float]
    base_voltages: List[float]
    slack_bus: List[str]
    unique_ids: List[str]


class LabelledArray(BaseModel):
    array: List[float]
    unique_ids: List[str]


def make_labelled_array(array, binary_mask):
    (integer_ids,) = np.nonzero(binary_mask)
    return LabelledArray(
        array=list(array[integer_ids]), unique_ids=list(map(str, integer_ids))
    )


def numpy_to_y_matrix(array):
    return [
        [Complex(real=element.real, imag=element.imag) for element in row]
        for row in array
    ]


def setup_sim(config: FeederConfig):
    sim = FeederSimulator(config)

    snapshot_run(dss)
    return sim


def get_true_phases(angle):
    if np.abs(angle-0)<0.2:
        return 0
    elif np.abs(angle-np.pi/3)<0.2:
        return np.pi/3
    elif np.abs(angle-2*np.pi/3)<0.2:
        return 2*np.pi/3
    elif np.abs(angle-3*np.pi/3)<0.2:
        return 3*np.pi/3
    elif np.abs(angle-(-np.pi/3))<0.2:
        return -np.pi/3
    elif np.abs(angle-(-2*np.pi/3))<0.2:
        return -2*np.pi/3
    elif np.abs(angle-(-3*np.pi/3))<0.2:
        return -3*np.pi/3
    else:
        print("error")


def go_cosim(sim, config: FeederConfig):

    deltat = 0.01
    fedinitstring = "--federates=1"

    print("Creating Federate Info")
    fedinfo = h.helicsCreateFederateInfo()
    h.helicsFederateInfoSetCoreName(fedinfo, config.name)
    h.helicsFederateInfoSetCoreTypeFromString(fedinfo, "zmq")
    h.helicsFederateInfoSetCoreInitString(fedinfo, fedinitstring)
    h.helicsFederateInfoSetTimeProperty(fedinfo, h.helics_property_time_delta, deltat)
    vfed = h.helicsCreateValueFederate(config.name, fedinfo)

    pub_voltages_real = h.helicsFederateRegisterPublication(vfed, "voltages_real", h.HELICS_DATA_TYPE_STRING, "")
    pub_voltages_imag = h.helicsFederateRegisterPublication(vfed, "voltages_imag", h.HELICS_DATA_TYPE_STRING, "")
    pub_powers_real = h.helicsFederateRegisterPublication(vfed, "powers_real", h.HELICS_DATA_TYPE_STRING, "")
    pub_powers_imag = h.helicsFederateRegisterPublication(vfed, "powers_imag", h.HELICS_DATA_TYPE_STRING, "")
    pub_topology = h.helicsFederateRegisterPublication(vfed, "topology", h.HELICS_DATA_TYPE_STRING, "")

    h.helicsFederateEnterExecutingMode(vfed)

    Y = sim.get_y_matrix()
    logger.debug("Eigenvalues and vectors")
    logger.debug(np.linalg.eig(Y.toarray()))
    y_matrix = numpy_to_y_matrix(Y.toarray())
    def get_phase(name):
        _, end = name.split('.')
        if end == '1':
            return 0
        elif end == '2':
            return -2*np.pi/3
        elif end == '3':
            return 2*np.pi/3
        else:
            raise Exception("Cannot parse name")

    phases = list(map(get_phase, sim._AllNodeNames))
    logger.debug("_Vbase_allnode")
    logger.debug(sim._Vbase_allnode)
    base_voltages = list(sim._Vbase_allnode)

    slack_bus = [
        sim._AllNodeNames[i] for i in range(
            sim._source_indexes[0], sim._source_indexes[-1] + 1
        )
    ]

    unique_ids = sim._AllNodeNames
    topology = Topology(
        y_matrix=y_matrix,
        phases=phases,
        base_voltages=base_voltages,
        slack_bus=slack_bus,
        unique_ids=unique_ids
    )

    pub_topology.publish(topology.json())

    snapshot_run(sim)

    granted_time = -1
    current_index = config.start_time_index
    for request_time in range(0, 100):
        while granted_time < request_time:
            granted_time = h.helicsFederateRequestTime(vfed, request_time)
        current_index+=1
        logger.info(f'Get Voltages and PQs at {current_index} {granted_time} {request_time}')


        pv_ = pv_df.loc[sim._simulation_step][0]

        sim.set_load_pq_timeseries(load_df)
        if granted_time <= 2:
            sim.set_gen_pq(pv_, pv_)
        sim.solve()

        feeder_voltages = sim.get_voltages_actual()
        PQ_node = sim.get_PQs()
        logger.debug("PQ")
        logger.debug(PQ_node)
        logger.debug("Calculated Power")
        errors = PQ_node + feeder_voltages * (Y.conjugate() @ feeder_voltages.conjugate()) / 1000
        power_balance = (feeder_voltages * (Y.conjugate() @ feeder_voltages.conjugate()) / 1000)
        logger.debug(power_balance)
        indices, = np.nonzero(np.abs(errors) > 1)
        logger.debug("Indices with error > 1")
        logger.debug(indices)
        logger.debug([sim._AllNodeNames[i] for i in indices])
        logger.debug("Power, Voltages, and Calculated Power at Indices")
        logger.debug(PQ_node[indices])
        logger.debug(feeder_voltages[indices])
        logger.debug(power_balance[indices])


        phases = list(map(get_true_phases, np.angle(feeder_voltages)))
        topology = Topology(
            y_matrix=y_matrix,
            phases=phases,
            base_voltages=base_voltages,
            slack_bus=slack_bus,
            unique_ids=unique_ids
        )
        pub_topology.publish(topology.json())

        print('Publish load ' + str(feeder_voltages.real[0]))
        pub_voltages_real.publish(LabelledArray(array=list(feeder_voltages.real), unique_ids=sim._AllNodeNames).json())
        pub_voltages_imag.publish(LabelledArray(array=list(feeder_voltages.imag), unique_ids=sim._AllNodeNames).json())
        pub_powers_real.publish(LabelledArray(array=list(PQ_node.real), unique_ids=sim._AllNodeNames).json())
        pub_powers_imag.publish(LabelledArray(array=list(PQ_node.imag), unique_ids=sim._AllNodeNames).json())

        sim.run_next()

    h.helicsFederateDisconnect(vfed)
    h.helicsFederateFree(vfed)
    h.helicsCloseLibrary()


class FeederCosimConfig(BaseModel):
    feeder_config: FeederConfig


def run():
    with open('static_inputs.json') as f:
        parameters = json.load(f)
    config = FeederConfig(**parameters)
    sim = setup_sim(config)
    go_cosim(sim, config)

if __name__ == '__main__':
    run()
