import logging
import helics as h
import opendssdirect as dss
import pandas as pd
import json
from dss_functions import snapshot_run
from FeederSimulator import FeederSimulator, FeederConfig
from pydantic import BaseModel
from typing import List, Tuple
import numpy as np
from datetime import datetime, timedelta
from gadal.gadal_types.data_types import (
        Complex,Topology,VoltagesReal,VoltagesImaginary,PowersReal,
        PowersImaginary, AdmittanceMatrix, VoltagesMagnitude, 
        VoltagesAngle, Injection, AdmittanceSparse
)

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.DEBUG)

test_se = False

def numpy_to_y_matrix(array):
    return [
        [(element.real, element.imag) for element in row]
        for row in array
    ]

def sparse_to_admittance_sparse(array, unique_ids):
    return AdmittanceSparse(
        from_equipment = [unique_ids[i] for i in array.row],
        to_equipment = [unique_ids[i] for i in array.col],
        admittance_list = [(data.real, data.imag) for data in array.data]
    )

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

    pub_voltages_real = h.helicsFederateRegisterPublication(vfed, "voltages_real", h.HELICS_DATA_TYPE_STRING, "")
    pub_voltages_imag = h.helicsFederateRegisterPublication(vfed, "voltages_imag", h.HELICS_DATA_TYPE_STRING, "")
    pub_voltages_magnitude = h.helicsFederateRegisterPublication(vfed, "voltages_magnitude", h.HELICS_DATA_TYPE_STRING, "")
    pub_powers_real = h.helicsFederateRegisterPublication(vfed, "powers_real", h.HELICS_DATA_TYPE_STRING, "")
    pub_powers_imag = h.helicsFederateRegisterPublication(vfed, "powers_imag", h.HELICS_DATA_TYPE_STRING, "")
    pub_topology = h.helicsFederateRegisterPublication(vfed, "topology", h.HELICS_DATA_TYPE_STRING, "")

    h.helicsFederateEnterExecutingMode(vfed)

    Y = sim.get_y_matrix()
    #logger.debug("Eigenvalues and vectors")
    #logger.debug(np.linalg.eig(Y.toarray()))
    #y_matrix = numpy_to_y_matrix(Y.toarray())
    unique_ids = sim._AllNodeNames

    if config.use_sparse_admittance:
        admittancematrix = sparse_to_admittance_sparse(
            Y,
            unique_ids
        )
    else:
        admittancematrix = AdmittanceMatrix(
            admittance_matrix= numpy_to_y_matrix(
                Y.toarray()
            ),
            ids=unique_ids
        )


    logger.debug("_Vbase_allnode")
    logger.debug(sim._Vbase_allnode)

    slack_bus = [
        sim._AllNodeNames[i] for i in range(
            sim._source_indexes[0], sim._source_indexes[-1] + 1
        )
    ]

    unique_ids = sim._AllNodeNames
    snapshot_run(sim)

    logger.debug("slack_bus")
    logger.debug(slack_bus)
    logger.debug("unique_ids")
    logger.debug(unique_ids)

   
    all_PQs = {}
    all_PQs['load'] = sim.get_PQs_load(static=True) # Return type is PQ_values, PQ_names, PQ_types all with same size
    all_PQs['pv'] = sim.get_PQs_pv(static=True)
    all_PQs['gen'] = sim.get_PQs_gen(static=True)
    all_PQs['cap'] = sim.get_PQs_cap(static=True)

    PQ_real = []
    PQ_imaginary = []
    PQ_names = []
    PQ_types = []
    for i in range(len(all_PQs['load'][0])):
        for key in all_PQs:
            if all_PQs[key][1][i] !='':
                PQ_real.append(-1*all_PQs[key][0][i].real) # injections are negative singe PQ values are positive
                PQ_imaginary.append(-1*all_PQs[key][0][i].imag) # injections are negative singe PQ values are positive
                PQ_names.append(all_PQs[key][1][i])
                PQ_types.append(all_PQs[key][2][i])
    power_real = PowersReal(ids = PQ_names, values = PQ_real, equipment_type = PQ_types)
    power_imaginary = PowersImaginary(ids = PQ_names, values = PQ_imaginary, equipment_type = PQ_types)
    injections = Injection(power_real=power_real, power_imaginary = power_imaginary)

    sim.solve(0,0)
    feeder_voltages = sim.get_voltages_actual()
    phases = list(map(get_true_phases, np.angle(feeder_voltages)))
    base_voltages = list(sim._Vbase_allnode)
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
        injections=injections,
        base_voltage_magnitudes=base_voltagemagnitude,
        slack_bus=slack_bus,
    )

    logger.info("Sending topology and saving to topology.json")
    with open(config.topology_output, "w") as f:
        f.write(topology.json())
    pub_topology.publish(topology.json())


    granted_time = -1
    current_hour = 0
    current_second = 0
    current_index = config.start_time_index
    for request_time in range(0, int(config.number_of_timesteps)):
        while granted_time < request_time:
            granted_time = h.helicsFederateRequestTime(vfed, request_time)

        logger.info('start time: '+str(datetime.now()))
        current_index+=1
        current_timestamp = datetime.strptime(config.start_date, '%Y-%m-%d %H:%M:%S') + timedelta(seconds = current_index*config.run_freq_sec)
        current_second+=config.run_freq_sec
        if current_second >=60*60:
            current_second = 0
            current_hour+=1
        logger.info(f'Get Voltages and PQs at {current_index} {granted_time} {request_time}')

        sim.solve(current_hour,current_second)

        feeder_voltages = sim.get_voltages_actual()
        all_PQs = {}
        all_PQs['load'] = sim.get_PQs_load(static=False) # Return type is PQ_values, PQ_names, PQ_types all with same size
        all_PQs['pv'] = sim.get_PQs_pv(static=False)
        all_PQs['gen'] = sim.get_PQs_gen(static=False)
        all_PQs['cap'] = sim.get_PQs_cap(static=False)
    
        PQ_injections_all = all_PQs['load'][0]+ all_PQs['pv'][0] + all_PQs['gen'][0]+ all_PQs['cap'][0]

        logger.debug("Feeder Voltages")
        logger.debug(feeder_voltages)
        logger.debug("PQ")
        logger.debug(PQ_injections_all)
        logger.debug("Calculated Power")
        Cal_power = feeder_voltages * (Y.conjugate() @ feeder_voltages.conjugate()) / 1000
        errors = PQ_injections_all + Cal_power
        sort_errors = np.sort(np.abs(errors))
        logger.debug("errors")
        logger.debug(errors)
        if np.any(sort_errors[:-3] > 1):
            raise ValueError('Power balance does not hold')
        PQ_injections_all[sim._source_indexes] = -Cal_power[sim._source_indexes]
        power_balance = (feeder_voltages * (Y.conjugate() @ feeder_voltages.conjugate()) / 1000)
        logger.debug(power_balance)
        indices, = np.nonzero(np.abs(errors) > 1)
        logger.debug("Indices with error > 1")
        logger.debug(indices)
        logger.debug([sim._AllNodeNames[i] for i in indices])
        logger.debug("Power, Voltages, and Calculated Power at Indices")
        logger.debug(PQ_injections_all[indices])
        logger.debug(feeder_voltages[indices])
        logger.debug(power_balance[indices])


        phases = list(map(get_true_phases, np.angle(feeder_voltages)))

        base_voltageangle = VoltagesAngle(
           values = phases,
           ids = unique_ids
        )

        logger.debug('Publish load ' + str(feeder_voltages.real[0]))
        voltage_magnitudes = np.abs(feeder_voltages.real + 1j* feeder_voltages.imag)
        pub_voltages_magnitude.publish(VoltagesMagnitude(values=list(voltage_magnitudes), ids=sim._AllNodeNames, time = current_timestamp).json())
        pub_voltages_real.publish(VoltagesReal(values=list(feeder_voltages.real), ids=sim._AllNodeNames, time = current_timestamp).json())
        pub_voltages_imag.publish(VoltagesImaginary(values=list(feeder_voltages.imag), ids=sim._AllNodeNames, time = current_timestamp).json())
        pub_powers_real.publish(PowersReal(values=list(PQ_injections_all.real), ids=sim._AllNodeNames, time = current_timestamp).json())
        pub_powers_imag.publish(PowersImaginary(values=list(PQ_injections_all.imag), ids=sim._AllNodeNames, time = current_timestamp).json())

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
