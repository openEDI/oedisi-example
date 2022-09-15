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
from gadal.gadal_types.data_types import Complex,Topology,VoltagesReal,VoltagesImaginary,PowersReal,PowersImaginary, AdmittanceMatrix, VoltagesMagnitude, VoltagesAngle

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)

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

    deltat = config.deltat
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

    pub_topology_flow = h.helicsFederateRegisterPublication(vfed, "topology_flow", h.HELICS_DATA_TYPE_STRING, "")
    logger.info(f'topology - flow matrix - publishing created')

    pub_taps_info = h.helicsFederateRegisterPublication(vfed, "tap_info", h.HELICS_DATA_TYPE_STRING, "")
    logger.info(f'taps info publishing created')

    pub_caps_info = h.helicsFederateRegisterPublication(vfed, "cap_info", h.HELICS_DATA_TYPE_STRING, "")
    logger.info(f'caps info publishing created')

    pub_flex_info = h.helicsFederateRegisterPublication(vfed, "flex_info", h.HELICS_DATA_TYPE_STRING, "")
    logger.info(f'flex loads info publishing created')

    pub_pv_info = h.helicsFederateRegisterPublication(vfed, "pv_info", h.HELICS_DATA_TYPE_STRING, "")
    logger.info(f'pv systems info publishing created')

    pub_powers_real = h.helicsFederateRegisterPublication(vfed, "powers_real", h.HELICS_DATA_TYPE_STRING, "")
    logger.info(f'real power publishing created')

    pub_powers_imag = h.helicsFederateRegisterPublication(vfed, "powers_imag", h.HELICS_DATA_TYPE_STRING, "")
    logger.info(f'imag power publishing created')

    pub_cap_powers_imag = h.helicsFederateRegisterPublication(vfed, "cap_powers_imag", h.HELICS_DATA_TYPE_STRING,
                                                                   "")
    logger.info(f'capacitor values publishing created')

    pub_pv_powers_real = h.helicsFederateRegisterPublication(vfed, "pv_powers_real", h.HELICS_DATA_TYPE_STRING, "")
    logger.info(f'real pv real power publishing created')

    pub_pv_powers_imag = h.helicsFederateRegisterPublication(vfed, "pv_powers_imag", h.HELICS_DATA_TYPE_STRING, "")
    logger.info(f'imag pv power publishing created')

    pub_tap_values = h.helicsFederateRegisterPublication(vfed, "tap_values", h.HELICS_DATA_TYPE_STRING, "")
    logger.info(f'tap values publishing created')

    opf_control = OPFControl(
        sub_powers_flex = register_sub_or_none("opf_flex_powers_real", "W"),
        sub_cap_powers_imag = register_sub_or_none("opf_cap_powers_imag", "Var"),
        sub_pv_powers_real = register_sub_or_none("opf_pv_powers_real", "W"),
        sub_pv_powers_imag = register_sub_or_none("opf_pv_powers_imag", "Var"),
        sub_tap_values = register_sub_or_none("opf_tap_values"),
    )
    h.helicsFederateEnterExecutingMode(vfed)
    logger.info(f'Federate Execution Mode Entered')


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

        y_line = numpy_to_y_matrix(sim._y_line)
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
    unique_ids = sim._AllNodeNames
    # topology = Topology(
    #     y_matrix=y_matrix,
    #     phases=phases,
    #     base_voltages=base_voltages,
    #     slack_bus=slack_bus,
    #     unique_ids=unique_ids
    # )

    topology_flow = Topology(
        y_matrix=y_line,
        phases=phases,
        base_voltages=base_voltages,
        slack_bus=slack_bus,
        unique_ids=unique_ids
    )
    taps_info = GenModelInfo(
        adj_matrix=sim._opf_reg_inc_matrix.tolist(),
        values=sim._opf_reg_taps_rated.tolist(),
        names=sim._opf_reg_order_names
    )
    caps_info = GenModelInfo(
        adj_matrix=sim._cap_load_inc_matrix.tolist(),
        values=sim._opf_cap_bank_rated.tolist(),
        names=sim._opf_cap_names.tolist()
    )
    flex_info = GenModelInfo(
        adj_matrix=sim._opf_flex_load_inc_matrix.tolist(),
        values=sim._opf_p_flex_load_rated.tolist(),
        names=sim._opf_p_flex_names
    )
    pv_info = PVModelInfo(
        adj_matrix=sim._opf_pv_inc_matrix.tolist(),
        p_values=sim._opf_pv_p_rated.tolist(),
        q_values=sim._opf_pv_q_rated.tolist(),
        s_values=sim._opf_pv_s_rated.tolist(),
        names=sim._opf_pv_names.tolist()
    )
    logger.info("Sending topology and saving to topology.json")
    with open("topology.json", "w") as topology_file:
        topology_file.write(topology.json())
    pub_topology.publish(topology.json())

    snapshot_run(sim)

    granted_time = -1
    current_hour = 0
    current_second = 0
    current_index = config.start_time_index
    granted_time = 0
    with open('nodenames.csv', 'w') as f:
        wr = csv.writer(f, dialect='excel')
        wr.writerows(sim._AllNodeNames)
    f.close()

    voltages_list = []
    active_power_loads_list = []
    reactive_power_loads_list = []
    active_power_pv_list = []
    reactive_power_pv_list = []
    taps_list = []
    cap_reactive_power_list = []
    for request_time in range(0, int(config.number_of_timesteps)):
        while granted_time < request_time:
            granted_time = h.helicsFederateRequestTime(vfed, request_time)

            if current_index == config.start_time_index:
                pub_topology.publish(topology.json())
                logger.info(f'Published Topology')
                pub_topology_flow.publish(topology_flow.json())
                logger.info(f'Published line flow topology')
                pub_flex_info.publish(flex_info.json())
                logger.info(f'Published Flex Load Adjacency Info')
                pub_taps_info.publish(taps_info.json())
                logger.info(f'Published Taps Adjacency Info')
                pub_caps_info.publish(caps_info.json())
                logger.info(f'Published Caps Adjacency Info')
                pub_pv_info.publish(pv_info.json())
                logger.info(f'Published PVSystems Adjacency Info')
                # pv_ = pv_df.loc[sim._simulation_step][0]
                # sim.set_load_pq_timeseries(oad_df)
                # sim.set_pv_pq(pv_, 0)
                # sim.solve()
                # sim.get_opf_system_vecs()
                sim.solve()
                sim.get_opf_system_vecs()

                feeder_voltages = sim._voltages
                active_power_loads = sim._p_Y
                reactive_power_loads = sim._q_Y
                cap_reactive_power = sim._q_cap_Y
                active_power_pv = sim._p_pV_Y
                reactive_power_pv = sim._q_pV_Y
                taps = sim._reg_taps
                sim.solve(current_hour,current_second)

                feeder_voltages = sim.get_voltages_actual()
                PQ_node = sim.get_PQs()
                logger.debug("Feeder Voltages")
                logger.debug(feeder_voltages)
                logger.debug("PQ")
                logger.debug(PQ_node)

                logger.info('Publish load ' + str(np.sum(active_power_loads)))

                pub_voltages_real.publish(
                    LabelledArray(array=list(feeder_voltages.real), unique_ids=sim._opf_node_order).json())
                logger.info(f'real voltages published')
                #logger.info(f'length: {len(feeder_voltages.real)}')
                logger.debug(f'values: {feeder_voltages.real.tolist()}')
                pub_voltages_imag.publish(
                    LabelledArray(array=list(feeder_voltages.imag), unique_ids=sim._opf_node_order).json())
                logger.info(f'voltages_imag published')

                self.pub_powers_real.publish(
                    LabelledArray(array=list(active_power_loads), unique_ids=sim._opf_node_order).json())
                logger.info(f'powers_real published')

                self.pub_powers_imag.publish(
                    LabelledArray(array=list(reactive_power_loads), unique_ids=sim._opf_node_order).json())
                logger.info(f'powers_imag published')

                pub_cap_powers_imag.publish(
                    LabelledArray(array=list(cap_reactive_power), unique_ids=sim._AllNodeNames).json())
                logger.info(f'cap_powers_imag published published')

                pub_pv_powers_real.publish(LabelledArray(array=list(active_power_pv), unique_ids=sim._AllNodeNames).json())
                logger.info(f'pv_powers_real published')
                logger.debug(f"{list(active_power_pv)}")

                pub_pv_powers_imag.publish(
                    LabelledArray(array=list(reactive_power_pv), unique_ids=sim._AllNodeNames).json())
                logger.info(f'pv_powers_imag published')
                logger.debug(f"{list(reactive_power_pv)}")

                pub_tap_values.publish(LabelledArray(array=list(taps), unique_ids=sim._opf_reg_order_names).json())
                logger.info(f'tap_values published')

                logger.debug("Calculated Power")
                Cal_power = feeder_voltages * (Y.conjugate() @ feeder_voltages.conjugate()) / 1000
                errors = PQ_node + Cal_power
                PQ_node[:3] = -Cal_power[:3]
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

        logger.debug('Publish load ' + str(feeder_voltages.real[0]))
        voltage_magnitudes = np.abs(feeder_voltages.real + 1j* feeder_voltages.imag)
        pub_voltages_magnitude.publish(VoltagesMagnitude(values=list(voltage_magnitudes), ids=sim._AllNodeNames, time = current_timestamp).json())
        pub_voltages_real.publish(VoltagesReal(values=list(feeder_voltages.real), ids=sim._AllNodeNames, time = current_timestamp).json())
        pub_voltages_imag.publish(VoltagesImaginary(values=list(feeder_voltages.imag), ids=sim._AllNodeNames, time = current_timestamp).json())
        pub_powers_real.publish(PowersReal(values=list(PQ_node.real), ids=sim._AllNodeNames, time = current_timestamp).json())
        pub_powers_imag.publish(PowersImaginary(values=list(PQ_node.imag), ids=sim._AllNodeNames, time = current_timestamp).json())

        if ((granted_time - 60) % (60 * 60)) == 0:
            set_powers_flex(sim, self.opf_control)
            set_caps(sim, self.opf_control)
            set_pv(sim, self.opf_control)
            set_feeder_taps(sim, self.opf_control)
            # set_feeder_caps(dss, cap_Q)
            # set_feeder_pvs(dss, pv_P, pv_Q)
            # set_feeder_taps(dss, tap_vals)
        logger.info('end time: '+str(datetime.now()))

    voltages_list.append(np.abs(feeder_voltages))
    active_power_loads_list.append(active_power_loads)
    reactive_power_loads_list.append(reactive_power_loads.transpose())
    active_power_pv_list.append(active_power_pv.transpose())
    reactive_power_pv_list.append(reactive_power_pv.transpose())
    taps_list.append(taps.transpose())
    cap_reactive_power_list.append(cap_reactive_power.transpose())
    logger.info(f'writing to csv file')
    f = open('voltages.csv', 'wb')
    np.savetxt(f, np.array(voltages_list), delimiter=",")
    f.close()
    f = open('active_power_loads.csv', 'wb')
    np.savetxt(f, np.array(active_power_loads_list), delimiter=",")
    f.close()
    f = open('reactive_power_loads.csv', 'wb')
    np.savetxt(f, np.array(reactive_power_loads_list), delimiter=",")
    f.close()
    f = open('active_power_pv.csv', 'wb')
    np.savetxt(f, np.array(active_power_pv_list), delimiter=",")
    f.close()
    f = open('reactive_power_pv.csv', 'wb')
    np.savetxt(f, np.array(reactive_power_pv_list), delimiter=",")
    f.close()
    f = open('taps.csv', 'wb')
    np.savetxt(f, np.array(taps_list), delimiter=",")
    f.close()
    f = open('cap_reactive_power.csv', 'wb')
    np.savetxt(f, np.array(cap_reactive_power_list), delimiter=",")
    f.close()
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
