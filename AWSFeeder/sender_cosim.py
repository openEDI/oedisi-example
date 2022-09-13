import logging
import helics as h
import opendssdirect as dss
import sys
import pandas as pd
import json
from dss_functions import snapshot_run
from FeederSimulator import FeederSimulator, FeederConfig
from pydantic import BaseModel
from typing import List, Tuple
import numpy as np
from datetime import datetime, timedelta
from gadal.gadal_types.data_types import Complex,Topology,VoltagesReal,VoltagesImaginary,PowersReal,PowersImaginary, AdmittanceMatrix, VoltagesMagnitude, VoltagesAngle

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)
#logger.setLevel(logging.DEBUG)

test_se = False


def numpy_to_y_matrix(array):
    return [
        [(element.real, element.imag) for element in row]
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
        logger.debug("error")


def go_cosim(sim, config: FeederConfig):

    deltat = 0.01
    fedinitstring = "--federates=1"

    logger.info("Creating Federate Info")
    fedinfo = h.helicsCreateFederateInfo()
    h.helicsFederateInfoSetCoreName(fedinfo, config.name)
    h.helicsFederateInfoSetCoreTypeFromString(fedinfo, "zmq")
    h.helicsFederateInfoSetCoreInitString(fedinfo, fedinitstring)
    h.helicsFederateInfoSetTimeProperty(fedinfo, h.helics_property_time_delta, deltat)
    vfed = h.helicsCreateValueFederate(config.name, fedinfo)

    pub_voltages_magnitude = h.helicsFederateRegisterPublication(vfed, "voltages_magnitude", h.HELICS_DATA_TYPE_STRING, "")
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
    base_voltages = list(sim._Vbase_allnode)

    slack_bus = [
        sim._AllNodeNames[i] for i in range(
            sim._source_indexes[0], sim._source_indexes[-1] + 1
        )
    ]

    unique_ids = sim._AllNodeNames

    logger.debug("y-matrix")
    logger.debug(y_matrix)
    logger.debug("phases")
    logger.debug(phases)
    logger.debug("base_voltages")
    logger.debug(base_voltages)
    logger.debug("slack_bus")
    logger.debug(slack_bus)
    logger.debug("unique_ids")
    logger.debug(unique_ids)
    
    
    admittancematrix = AdmittanceMatrix(
        admittance_matrix = y_matrix,
        ids = unique_ids
    )

    base_voltagemagnitude = VoltagesMagnitude(
           values = [abs(i) for i in base_voltages],
           ids = unique_ids
    )

    base_voltageangle = VoltagesAngle(
           values = phases,
           ids = unique_ids
    )

    topology = Topology(
        admittance=admittancematrix,
        base_voltage_angles=base_voltageangle,
        injections={},
        base_voltage_magnitudes=base_voltagemagnitude,
        slack_bus=slack_bus,
    )


    logger.info("Sending topology and saving to topology.json")
    pub_topology.publish(topology.json())
    with open("topology.json", "w") as f:
        f.write(topology.json())

    snapshot_run(sim)

    granted_time = -1
    current_index = 0
    current_hour = 0
    current_second = 0
    for request_time in range(0, config.number_of_timesteps):
        while granted_time < request_time:
            granted_time = h.helicsFederateRequestTime(vfed, request_time)
        logger.info('start time: '+str(datetime.now()))
        current_index+=1
        current_timestamp = datetime.strptime(sim._start_date, '%Y-%m-%d %H:%M:%S') + timedelta(minutes = current_index*15)
        current_second+=15*60
        if current_second >=60*60:
            current_second = 0
            current_hour+=1
        logger.info(f'Get Voltages and PQs at {current_index} {granted_time} {request_time}')

        sim.solve(current_hour,current_second)

        feeder_voltages = sim.get_voltages_actual()
        PQ_node = sim.get_PQs()
        logger.debug("Feeder Voltages")
        logger.debug(feeder_voltages)
        logger.debug("PQ")
        logger.debug(PQ_node)
        logger.debug("Calculated Power")
        errors = PQ_node + feeder_voltages * (Y.conjugate() @ feeder_voltages.conjugate()) / 1000
        logger.debug("errors")
        logger.debug(errors)
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
        base_voltageangle = VoltagesAngle(
                values = phases,
                ids = unique_ids
        )
        topology = Topology(
            admittance=admittancematrix,
            base_voltage_angles=base_voltageangle,
            injections={},
            base_voltage_magnitudes=base_voltagemagnitude,
            slack_bus=slack_bus,
        )
        pub_topology.publish(topology.json())

        logger.info('Publish load ' + str(feeder_voltages.real[0]))
        voltage_magnitudes = np.abs(feeder_voltages.real + 1j* feeder_voltages.imag)
        pub_voltages_magnitude.publish(VoltagesMagnitude(values=list(voltage_magnitudes), ids=sim._AllNodeNames, time = current_timestamp).json())
        pub_voltages_real.publish(VoltagesReal(values=list(feeder_voltages.real), ids=sim._AllNodeNames, time = current_timestamp).json())
        pub_voltages_imag.publish(VoltagesImaginary(values=list(feeder_voltages.imag), ids=sim._AllNodeNames, time = current_timestamp).json())
        pub_powers_real.publish(PowersReal(values=list(PQ_node.real), ids=sim._AllNodeNames, time = current_timestamp).json())
        pub_powers_imag.publish(PowersImaginary(values=list(PQ_node.imag), ids=sim._AllNodeNames, time = current_timestamp).json())
        logger.info('end time: '+str(datetime.now()))


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
