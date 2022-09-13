import logging
import helics as h
import opendssdirect as dss
import pandas as pd
import json
# from opf_dss_functions import set_feeder_flex_load
from dss_functions import snapshot_run
from FeederSimulator import FeederSimulator, FeederConfig

from pydantic import BaseModel
from typing import List, Optional
import numpy as np

from dataclasses import dataclass

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.DEBUG)

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

#
# class to support passing of adjacency matrix of the controllable device location in the network
class GenModelInfo(BaseModel):
    adj_matrix: List[List[float]]
    values: List[float]
    names: List[str]

class PVModelInfo(BaseModel):
    adj_matrix: List[List[float]]
    p_values: List[float]
    q_values: List[float]
    s_values: List[float]
    names: List[str]

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

# @dataclass(kw_only=True)
@dataclass()
class OPFControl:
    sub_powers_flex: Optional[h.HelicsInput]
    sub_cap_powers_imag: Optional[h.HelicsInput]
    sub_pv_powers_real: Optional[h.HelicsInput]
    sub_pv_powers_imag: Optional[h.HelicsInput]
    sub_tap_values: Optional[h.HelicsInput]


def set_powers_flex(sim, opf_control):
    if opf_control.sub_powers_flex is not None:
        sim.opf_flex_loads_vars = LabelledArray.parse_obj(opf_control.sub_powers_flex.json)
        logger.info(f'Subscribed Flexible Power Curtailment Received from OPF')
        sim.set_feeder_flex_load()

def set_caps(sim, opf_control):
    if opf_control.sub_cap_powers_imag is not None:
        sim.opf_caps_vars = LabelledArray.parse_obj(opf_control.sub_cap_powers_imag.json)
        logger.info(f'Subscribed Cap Powers Imag Received from OPF')
        sim.set_feeder_caps()

def set_pv(sim, opf_control):
    if opf_control.sub_pv_powers_real is not None and opf_control.sub_pv_powers_imag is not None:
        sim.opf_pv_p_vars = LabelledArray.parse_obj(opf_control.sub_pv_powers_real.json)
        sim.opf_pv_q_vars = LabelledArray.parse_obj(opf_control.sub_pv_powers_imag.json)
        logger.info(f'Subscribed feeder pv powers received from OPF')
        sim.set_feeder_pvs()

def set_feeder_taps(sim, opf_control):
    if opf_control.sub_tap_values is not None:
        sim.opf_tap_vars = LabelledArray.parse_obj(opf_control.sub_tap_values.json)
        logger.info(f'Subscribed tap values received from OPF')
        sim.set_feeder_taps()


class SimulationFederate:
    def __init__(self, config: FeederConfig, input_mapping):
        self.sim = setup_sim(config)
        sim = self.sim
        #load_df = pd.read_csv(config.load_file)
        #load_df.columns = ['time']+sim._load_names

        data = pd.read_csv(config.load_file)

        # # specific loads
        #     ld_df = data.iloc[:, data.columns.str.contains('ld') == True]

        sub_p_norm = (data["sub_p"].values)/np.max(abs(data["sub_p"].values))
        sub_q_norm = (np.abs(data["sub_q"].values))/np.max(abs(data["sub_q"].values))

        load_kW_matrix = np.repeat(sim._opf_loads_df['kW'].values, [len(sub_p_norm)]).reshape(
            (len(sim._opf_loads_df['kW'].values), len(sub_p_norm))).transpose()
        load_kVar_matrix = np.repeat(sim._opf_loads_df['kvar'].values, [len(sub_p_norm)]).reshape(
            (len(sim._opf_loads_df['kW'].values), len(sub_q_norm))).transpose()


        # load_df_kW_values_matrix = np.tile(load_df_values, (len(sim._load_names), 1)).transpose()
        load_vals = load_kW_matrix*sub_p_norm[:, None] + 1j*load_kVar_matrix*sub_q_norm[:, None]
        self.load_df = pd.DataFrame(data=load_vals, index=data.index,
                            columns=sim._load_names).fillna(0)
        # load_df_kW = pd.DataFrame(data=load_kW_matrix*sub_p_norm[:, None], index=data.index,
        #                        columns=sim._load_names).fillna(0)
        # load_df_kVar = pd.DataFrame(data=load_kVar_matrix*sub_q_norm[:, None], index=data.index,
        #                        columns=sim._load_names).fillna(0)

        # sim.set_load_pq_timeseries(load_df)

        # sim.set_load_pq_timeseries(load_df_kW, load_df_kVar)

        # pv_df = pd.read_csv(config.pv_file, header=None, names=['pv'])
        #    pv_df['pv'] = pv_df['pv'] / pv_df['pv'].max()
        self.pv_df = pd.DataFrame(data=-data['pvP']/np.max(abs(data['pvP'].values)), index=data.index)

        self.input_mapping = input_mapping
        self.initialize_federate(config)

    def initialize_federate(self, config):
        fedinitstring = "--federates=1"

        logger.info(f'Creating Feeder Federate Info')
        fedinfo = h.helicsCreateFederateInfo()
        h.helicsFederateInfoSetCoreName(fedinfo, config.name)
        h.helicsFederateInfoSetCoreTypeFromString(fedinfo, "zmq")
        h.helicsFederateInfoSetCoreInitString(fedinfo, fedinitstring)
        h.helicsFederateInfoSetTimeProperty(fedinfo, h.helics_property_time_delta, config.deltat)
        vfed = h.helicsCreateValueFederate(config.name, fedinfo)
        self.federate = vfed
        logger.info(f'FEEDER Federate Created')

        self.pub_topology = h.helicsFederateRegisterPublication(vfed, "topology", h.HELICS_DATA_TYPE_STRING, "")
        logger.info(f'topology - y_matrix - publishing created')

        self.pub_topology_flow = h.helicsFederateRegisterPublication(vfed, "topology_flow", h.HELICS_DATA_TYPE_STRING, "")
        logger.info(f'topology - flow matrix - publishing created')

        self.pub_taps_info = h.helicsFederateRegisterPublication(vfed, "tap_info", h.HELICS_DATA_TYPE_STRING, "")
        logger.info(f'taps info publishing created')

        self.pub_caps_info = h.helicsFederateRegisterPublication(vfed, "cap_info", h.HELICS_DATA_TYPE_STRING, "")
        logger.info(f'caps info publishing created')

        self.pub_flex_info = h.helicsFederateRegisterPublication(vfed, "flex_info", h.HELICS_DATA_TYPE_STRING, "")
        logger.info(f'flex loads info publishing created')

        self.pub_pv_info = h.helicsFederateRegisterPublication(vfed, "pv_info", h.HELICS_DATA_TYPE_STRING, "")
        logger.info(f'pv systems info publishing created')

        self.pub_voltages_real = h.helicsFederateRegisterPublication(vfed, "voltages_real", h.HELICS_DATA_TYPE_STRING, "")
        logger.info(f'real voltage publishing created')

        self.pub_voltages_imag = h.helicsFederateRegisterPublication(vfed, "voltages_imag", h.HELICS_DATA_TYPE_STRING, "")
        logger.info(f'imag voltage publishing created')

        self.pub_powers_real = h.helicsFederateRegisterPublication(vfed, "powers_real", h.HELICS_DATA_TYPE_STRING, "")
        logger.info(f'real power publishing created')

        self.pub_powers_imag = h.helicsFederateRegisterPublication(vfed, "powers_imag", h.HELICS_DATA_TYPE_STRING, "")
        logger.info(f'imag power publishing created')

        self.pub_cap_powers_imag = h.helicsFederateRegisterPublication(vfed, "cap_powers_imag", h.HELICS_DATA_TYPE_STRING, "")
        logger.info(f'capacitor values publishing created')

        self.pub_pv_powers_real = h.helicsFederateRegisterPublication(vfed, "pv_powers_real", h.HELICS_DATA_TYPE_STRING, "")
        logger.info(f'real pv real power publishing created')

        self.pub_pv_powers_imag = h.helicsFederateRegisterPublication(vfed, "pv_powers_imag", h.HELICS_DATA_TYPE_STRING, "")
        logger.info(f'imag pv power publishing created')

        self.pub_tap_values = h.helicsFederateRegisterPublication(vfed, "tap_values", h.HELICS_DATA_TYPE_STRING, "")
        logger.info(f'tap values publishing created')

        self.opf_control = OPFControl(
            sub_powers_flex = self.register_sub_or_none("opf_flex_powers_real", "W"),
            sub_cap_powers_imag = self.register_sub_or_none("opf_cap_powers_imag", "Var"),
            sub_pv_powers_real = self.register_sub_or_none("opf_pv_powers_real", "W"),
            sub_pv_powers_imag = self.register_sub_or_none("opf_pv_powers_imag", "Var"),
            sub_tap_values = self.register_sub_or_none("opf_tap_values"),
        )
        h.helicsFederateEnterExecutingMode(vfed)
        logger.info(f'Federate Execution Mode Entered')


    def register_sub_or_none(self, name, unit="-"):
        if name in self.input_mapping:
            sub = self.federate.register_subscription(
                self.input_mapping[name], unit)
            logger.info(f'{name} at {self.input_mapping[name]} values subscription created')
            return sub
        else:
            return None


    def go_cosim(self):
        sim = self.sim
        Y = sim.get_y_matrix()
        # sim.get_opf_system_vecs()

        logger.debug("Eigenvalues and vectors")
        logger.debug(np.linalg.eig(Y.toarray()))
        y_matrix = numpy_to_y_matrix(Y.toarray())

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
        topology = Topology(
            y_matrix=y_matrix,
            phases=phases,
            base_voltages=base_voltages,
            slack_bus=slack_bus,
            unique_ids=unique_ids
        )

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

        #TODO: decide time steps and time intervals
        minutes = 30

        total_interval = int(minutes*60 + 10)
        current_index = config.start_time_index
        sim._simulation_step = current_index
        logger.info(f'HELICS PROPERTY TIME PERIOD {h.HELICS_PROPERTY_TIME_PERIOD}')
        update_interval = int(h.helicsFederateGetTimeProperty(self.federate, h.HELICS_PROPERTY_TIME_PERIOD))
        logger.info(f'update interval DSS Fed at start {update_interval}')

        granted_time = 0
        # for request_time in range(0, 100):
        while granted_time < total_interval:
            #logger.info(f'update interval DSS Fed {update_interval}')
            #logger.info(f'previous granted time DSS Fed {granted_time}')
            requested_time = (granted_time+update_interval)
            logger.info(f'requested time DSS Fed before HELICS {requested_time}')

            granted_time = h.helicsFederateRequestTime(self.federate, requested_time)
            logger.info(f'actual granted time to DSS Fed by HELICS {granted_time}')
            logger.info(f'Simulation Index {current_index}')
            logger.info(f"simulation step {sim._simulation_step}")
            # pub_topology.publish(topology.json())
            # logger.info(f'Published Topology')
            # snapshot_run(sim)

            if current_index == config.start_time_index:
                self.pub_topology.publish(topology.json())
                logger.info(f'Published Topology')
                self.pub_topology_flow.publish(topology_flow.json())
                logger.info(f'Published line flow topology')
                self.pub_flex_info.publish(flex_info.json())
                logger.info(f'Published Flex Load Adjacency Info')
                self.pub_taps_info.publish(taps_info.json())
                logger.info(f'Published Taps Adjacency Info')
                self.pub_caps_info.publish(caps_info.json())
                logger.info(f'Published Caps Adjacency Info')
                self.pub_pv_info.publish(pv_info.json())
                logger.info(f'Published PVSystems Adjacency Info')
                pv_ = self.pv_df.loc[sim._simulation_step][0]
                sim.set_load_pq_timeseries(self.load_df)
                sim.set_pv_pq(pv_, 0)
                sim.solve()
                sim.get_opf_system_vecs()

                feeder_voltages = sim._voltages
                active_power_loads = sim._p_Y
                reactive_power_loads = sim._q_Y
                cap_reactive_power = sim._q_cap_Y
                active_power_pv = sim._p_pV_Y
                reactive_power_pv = sim._q_pV_Y
                taps = sim._reg_taps

            else:
                sim._simulation_step = int(current_index/(60*15)) # every fifteen minutes
                logger.info(f'Get Voltages and PQs at {granted_time} {requested_time}')

                #if (current_index % 60*15) == 0: # change value every 60 seconds.
                pv_ = self.pv_df.loc[sim._simulation_step][0]
                sim.set_load_pq_timeseries(self.load_df)
            # if granted_time <= 2:
                sim.set_gen_pq(pv_, 0)
                sim.solve()
                sim.get_opf_system_vecs()

                feeder_voltages = sim._voltages
                active_power_loads = sim._p_Y
                reactive_power_loads = sim._q_Y
                cap_reactive_power = sim._q_cap_Y
                active_power_pv = sim._p_pV_Y
                reactive_power_pv = sim._q_pV_Y
                taps = sim._reg_taps
                # logger.info(f"{taps}")
                # TODO: previous ids were sim._AllNodeNames - check its imapct on the pub/sub
                logger.info('Publish load ' + str(np.sum(active_power_loads)))

                self.pub_voltages_real.publish(
                    LabelledArray(array=list(feeder_voltages.real), unique_ids=sim._opf_node_order).json())
                logger.info(f'real voltages published')
                #logger.info(f'length: {len(feeder_voltages.real)}')
                logger.debug(f'values: {feeder_voltages.real.tolist()}')
                self.pub_voltages_imag.publish(
                    LabelledArray(array=list(feeder_voltages.imag), unique_ids=sim._opf_node_order).json())
                logger.info(f'voltages_imag published')

                self.pub_powers_real.publish(
                    LabelledArray(array=list(active_power_loads), unique_ids=sim._opf_node_order).json())
                logger.info(f'powers_real published')

                self.pub_powers_imag.publish(
                    LabelledArray(array=list(reactive_power_loads), unique_ids=sim._opf_node_order).json())
                logger.info(f'powers_imag published')

                self.pub_cap_powers_imag.publish(
                    LabelledArray(array=list(cap_reactive_power), unique_ids=sim._AllNodeNames).json())
                logger.info(f'cap_powers_imag published published')

                self.pub_pv_powers_real.publish(LabelledArray(array=list(active_power_pv), unique_ids=sim._AllNodeNames).json())
                logger.info(f'pv_powers_real published')
                logger.debug(f"{list(active_power_pv)}")

                self.pub_pv_powers_imag.publish(
                    LabelledArray(array=list(reactive_power_pv), unique_ids=sim._AllNodeNames).json())
                logger.info(f'pv_powers_imag published')
                logger.debug(f"{list(reactive_power_pv)}")

                self.pub_tap_values.publish(LabelledArray(array=list(taps), unique_ids=sim._opf_reg_order_names).json())
                logger.info(f'tap_values published')

                try:
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
                except:  # TODO: Use actual error message
                    logger.info('Error in conjugate stuff')

                if (current_index % 60) == 0:
                    set_powers_flex(sim, self.opf_control)
                    set_caps(sim, self.opf_control)
                    set_pv(sim, self.opf_control)
                    set_feeder_taps(sim, self.opf_control)
                    # set_feeder_caps(dss, cap_Q)
                    # set_feeder_pvs(dss, pv_P, pv_Q)
                    # set_feeder_taps(dss, tap_vals)

            current_index+=1
            sim.run_next()
        self.finalize_federate()

    def finalize_federate(self):
        print(f"disconnecting federate")
        h.helicsFederateDisconnect(self.federate)
        h.helicsFederateFree(self.federate)
        print(f"feeder disconnected")
        h.helicsCloseLibrary()


if __name__ == '__main__':
    logger.debug(f'in run-->sender_cosim.py')

    with open('static_inputs.json') as f:
        logger.debug(f'opening static_inputs.json')
        parameters = json.load(f)
    with open("input_mapping.json") as f:
        input_mapping = json.load(f)

    config = FeederConfig(**parameters)
    logger.debug(f"Creating HELICS federate")
    sim = SimulationFederate(config, input_mapping)
    logger.debug(f"Running go_sim loop")
    sim.go_cosim()
    logger.debug(f"Out of go_sim loop")
