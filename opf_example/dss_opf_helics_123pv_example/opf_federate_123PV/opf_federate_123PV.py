"""
COPYRIGHT: Battelle Memorial, Pacific Northwest National Laboratory
This is following SGIDAL Working Example Template

This is a simple OPF federate that gets information from the grid Federate:
- It subscribes to:
    - Topology
    - Voltages
    - Powers
    - Tap Setpoints
    - Cap Bank Setpoints
- It publishes
    - load shedding setpoint
    - Tap regulator setpoints
    - Cap Bank Setpoints

@author: Sarmad Hanif
sarmad.hanif@pnnl.gov
"""
import helics as h
import logging
import numpy as np
from pydantic import BaseModel
from typing import List
import json
import opf_grid_utility_scripts
logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)




class Complex(BaseModel):
    "Pydantic model for complex values with json representation"
    real: float
    imag: float


class Topology(BaseModel):
    "All necessary data for state estimator run"
    y_matrix: List[List[Complex]]
    phases: List[float]
    base_voltages: List[float]
    slack_bus: List[str]
    unique_ids: List[str]


class GenModelInfo(BaseModel):
    adj_matrix: List[List[float]]
    values: List[float]
    names: List[str]

class PVModelInfo(BaseModel):
    adj_matrix: List[List[float]]
    p_values: List[float]
    q_values: List[float]
    names: List[str]


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


class OptimalPowerFlowFederate:
    "Optimal Power Flow Federate. Wraps OPF with Pubs and Subs"

    def __init__(self, config, input_mapping):
        "Initializes federate with name and remaps input into subscriptions"


        # Create Federate Info object that describes the federate properties #
        fedinfo = h.helicsCreateFederateInfo()

        fedinfo.core_name = config['name']
        fedinfo.core_type = h.HELICS_CORE_TYPE_ZMQ
        fedinfo.core_init = "--federates=1"
        h.helicsFederateInfoSetTimeProperty(
            fedinfo, h.helics_property_time_delta, config['deltat']
        )

        self.opf_fed = h.helicsCreateValueFederate(config['name'], fedinfo)
        logger.info(f"OPF federate created - registering subscriptions")

        # Register the subsciption #
        self.sub_topology = self.opf_fed.register_subscription(
            input_mapping["topology"], ""
        )
        logger.info(f"topology subscribed")

        self.sub_topology_flow = self.opf_fed.register_subscription(
            input_mapping["topology_flow"], ""
        )
        logger.info(f"topology_flow subscribed")

        self.sub_tap_info = self.opf_fed.register_subscription(
            input_mapping["tap_info"], ""
        )
        logger.info(f"tap_info subscribed")

        self.sub_cap_info = self.opf_fed.register_subscription(
            input_mapping["cap_info"], ""
        )
        logger.info(f"cap_info subscribed")

        self.sub_flex_info = self.opf_fed.register_subscription(
            input_mapping["flex_info"], ""
        )
        logger.info(f"flex_info subscribed")

        self.sub_pv_info = self.opf_fed.register_subscription(
            input_mapping["pv_info"], ""
        )
        logger.info(f"pv_info subscribed")


        self.sub_voltages_real = self.opf_fed.register_subscription(
            input_mapping["voltages_real"], "V"
        )
        logger.info(f"real voltages subscribed")

        self.sub_voltages_imag = self.opf_fed.register_subscription(
            input_mapping["voltages_imag"], "V"
        )
        logger.info(f"imaginary voltages subscribed")

        self.sub_powers_real = self.opf_fed.register_subscription(
            input_mapping["powers_real"], "W"
        )
        logger.info(f"active power load subscribed")

        self.sub_powers_imag = self.opf_fed.register_subscription(
            input_mapping["powers_imag"], "Var"
        )
        logger.info(f"reactive power load subscribed")

        self.sub_cap_powers_imag = self.opf_fed.register_subscription(
            input_mapping["cap_powers_imag"], "Var"
        )
        logger.info(f"cap_powers_imag subscribed")

        self.sub_pv_powers_real = self.opf_fed.register_subscription(
            input_mapping["pv_powers_real"], "W"
        )
        logger.info(f"pv_powers_real power subscribed")

        self.sub_pv_powers_imag = self.opf_fed.register_subscription(
            input_mapping["pv_powers_imag"], "Var"
        )
        logger.info(f"sub_pv_powers_imag power subscribed")

        self.sub_tap_values = self.opf_fed.register_subscription(
            input_mapping["tap_values"], ""
        )
        logger.info(f"sub_tap_values subscribed")

        # publishing to feeder
        self.pub_opf_flex_powers_real = h.helicsFederateRegisterPublication(self.opf_fed, "pub_opf_flex_powers_real",
                                                                            h.HELICS_DATA_TYPE_STRING, "")
        logger.info(f'real power to be published by OPF back to Feeder created')

        self.pub_opf_cap_powers_imag = h.helicsFederateRegisterPublication(self.opf_fed, "pub_opf_cap_powers_imag",
                                                                      h.HELICS_DATA_TYPE_STRING,
                                                                      "")
        logger.info(f'capacitor to be published by OPF back to Feeder created')

        self.pub_opf_pv_powers_real = h.helicsFederateRegisterPublication(self.opf_fed, "pub_opf_pv_powers_real",
                                                                     h.HELICS_DATA_TYPE_STRING, "")
        logger.info(f'real pv real power to be published by OPF back to Feeder created')

        self.pub_opf_pv_powers_imag = h.helicsFederateRegisterPublication(self.opf_fed, "pub_opf_pv_powers_imag",
                                                                     h.HELICS_DATA_TYPE_STRING, "")
        logger.info(f'imag pv power to be published by OPF back to Feeder created')

        self.pub_opf_tap_values = h.helicsFederateRegisterPublication(self.opf_fed, "pub_opf_tap_values",
                                                                 h.HELICS_DATA_TYPE_STRING, "")
        logger.info(f'tap values to be published by OPF back to Feeder created')


    def go_opf(self):
        logger.info(f'inside Run OPF Loop')

        "Enter execution and exchange data"
        # Enter execution mode #
        self.opf_fed.enter_executing_mode()
        logger.info("Entering execution mode")

        # seconds = 60
        minutes = 30
        total_interval = int(minutes*60 + 10)
        logger.info(f'HELICS PROPERTY TIME PERIOD {h.HELICS_PROPERTY_TIME_PERIOD}')
        update_interval = int(h.helicsFederateGetTimeProperty(self.opf_fed, h.HELICS_PROPERTY_TIME_PERIOD))
        logger.info(f'update interval OPF Fed at start {update_interval}')

        # running a second later after DSS Federate
        initial_time = 1
        logger.debug(f'Requesting initial time {initial_time}')
        granted_time = h.helicsFederateRequestTime(self.opf_fed, initial_time)
        logger.info(f'beginning granted time OPF Fed {granted_time}')

        # for request_time in range(0, 100):
        while granted_time < total_interval:
            logger.info(f'update interval OPF Fed {update_interval}')
            logger.info(f'previous granted time OPF Fed {granted_time}')
            requested_time = (granted_time + update_interval)
            logger.info(f'time requested by OPF Fed from HELICS {requested_time}')

            granted_time = h.helicsFederateRequestTime(self.opf_fed, requested_time)
            logger.info(f'actual granted time to OPF Fed by HELICS {granted_time}')

            # logger.info(f'While loop with granted time: {granted_time} requested time: {requested_time}')

            if int(granted_time-2) % (60*15) == 0:
                # 2 seconds delay
                # initialization - this could be done before the loop starts too.
                if granted_time == 2: # first time ever, we'd need to load up all vectors and matrices
                    self.topology = Topology.parse_obj(self.sub_topology.json)
                    logger.info(f'Subscribed Topology Received')

                    self.topology_flow = Topology.parse_obj(self.sub_topology_flow.json)
                    logger.info(f"Subscribed topology flow received")

                    self.tap_info = GenModelInfo.parse_obj(self.sub_tap_info.json)
                    logger.info(f"tap_info received")

                    self.cap_info = GenModelInfo.parse_obj(self.sub_cap_info.json)
                    logger.info(f"cap_info received")

                    self.flex_info = GenModelInfo.parse_obj(self.sub_flex_info.json)
                    logger.info(f"flex_info received")

                    self.pv_info = PVModelInfo.parse_obj(self.sub_pv_info.json)
                    logger.info(f"pv_info received")

                    # dynamic states, which may change
                    self.voltages_real = LabelledArray.parse_obj(self.sub_voltages_real.json)
                    logger.info(f'Subscribed REAL Voltages Received')

                    self.voltages_imag = LabelledArray.parse_obj(self.sub_voltages_imag.json)
                    logger.info(f'Subscribed Imag Voltages Received')

                    self.powers_P = LabelledArray.parse_obj(self.sub_powers_real.json)
                    logger.info(f'Subscribed Active Power Received')

                    self.powers_Q = LabelledArray.parse_obj(self.sub_powers_imag.json)
                    logger.info(f'Subscribed Reactive Power Received')

                    self.cap_Q = LabelledArray.parse_obj(self.sub_cap_powers_imag.json)
                    logger.info(f"cap_powers_imag received")

                    self.pv_P = LabelledArray.parse_obj(self.sub_pv_powers_real.json)
                    logger.info(f"sub_pv_powers_real power received")

                    self.pv_Q = LabelledArray.parse_obj(self.sub_pv_powers_imag.json)
                    logger.info(f"sub_pv_powers_imag  imag received")

                    self.tap_vals = LabelledArray.parse_obj(self.sub_tap_values.json)
                    logger.info(f"sub_tap_values received")
                    opf_grid_utility_scripts.unpack_fdrvals_mats(self)
                    opf_grid_utility_scripts.unpack_fdrvals_vecs(self)

                    logger.info(f"all grid quantities received")
                    logger.info(f"formulating the linear model")
                    opf_grid_utility_scripts.get_sens_matrices(self)
                    opf_grid_utility_scripts.solve_central_optimization(self)
                    logger.info("optimization complete passing the optimized variable setpoints to grid federate")


                    self.pub_opf_flex_powers_real.publish(
                        LabelledArray(array=list(self.p_flex_load_var_opti), unique_ids=self.flex_info.names).json())
                    logger.info(f'pub_powers_real published to feeder')

                    self.pub_opf_cap_powers_imag.publish(
                        LabelledArray(array=list(self.cap_value_opti), unique_ids=self.cap_info.names).json())
                    logger.info(f'pub_cap_powers_imag published published to feeder')

                    self.pub_opf_pv_powers_real.publish(
                        LabelledArray(array=list(self.p_pv_opti), unique_ids=self.pv_info.names).json())
                    logger.info(f'pub_pv_powers_real published to feeder')

                    self.pub_opf_pv_powers_imag.publish(
                        LabelledArray(array=list(self.q_pv_opti), unique_ids=self.pv_info.names).json())
                    logger.info(f'pub_pv_powers_imag published to feeder')

                    logger.info(f"pv names {self.pv_info.names}")
                    logger.info(f"pv p vals {self.p_pv_opti}")
                    logger.info(f"pv q vals {self.q_pv_opti}")

                    self.pub_opf_tap_values.publish(
                        LabelledArray(array=list(self.xmer_value_opti), unique_ids=self.tap_info.names).json())
                    logger.info(f'pub_tap_values published to feeder')

                else: # dynamic
                    # dynamic states, coming every 60 seconds or some predefined intervals
                    self.voltages_real = LabelledArray.parse_obj(self.sub_voltages_real.json)
                    logger.info(f'Subscribed REAL Voltages Received')

                    self.voltages_imag = LabelledArray.parse_obj(self.sub_voltages_imag.json)
                    logger.info(f'Subscribed Imag Voltages Received')

                    self.powers_P = LabelledArray.parse_obj(self.sub_powers_real.json)
                    logger.info(f'Subscribed Active Power Received')

                    self.powers_Q = LabelledArray.parse_obj(self.sub_powers_imag.json)
                    logger.info(f'Subscribed Reactive Power Received')

                    self.cap_Q = LabelledArray.parse_obj(self.sub_cap_powers_imag.json)
                    logger.info(f"cap_powers_imag received")

                    self.pv_P = LabelledArray.parse_obj(self.sub_pv_powers_real.json)
                    logger.info(f"sub_pv_powers_real power received")

                    self.pv_Q = LabelledArray.parse_obj(self.sub_pv_powers_imag.json)
                    logger.info(f"sub_pv_powers_imag  imag received")

                    self.tap_vals = LabelledArray.parse_obj(self.sub_tap_values.json)
                    logger.info(f"sub_tap_values received")

                    opf_grid_utility_scripts.unpack_fdrvals_vecs(self)
                    logger.info(f"formulating the linear model")
                    opf_grid_utility_scripts.get_sens_matrices(self)
                    opf_grid_utility_scripts.solve_central_optimization(self)

                    self.pub_opf_flex_powers_real.publish(
                        LabelledArray(array=list(self.p_flex_load_var_opti), unique_ids=self.flex_info.names).json())
                    logger.info(f'pub_powers_real published to feeder')

                    self.pub_opf_cap_powers_imag.publish(
                        LabelledArray(array=list(self.cap_value_opti), unique_ids=self.cap_info.names).json())
                    logger.info(f'pub_cap_powers_imag published published to feeder')

                    self.pub_opf_pv_powers_real.publish(
                        LabelledArray(array=list(self.p_pv_opti), unique_ids=self.pv_info.names).json())
                    logger.info(f'pub_pv_powers_real published to feeder')

                    self.pub_opf_pv_powers_imag.publish(
                        LabelledArray(array=list(self.q_pv_opti), unique_ids=self.pv_info.names).json())
                    logger.info(f'pub_pv_powers_imag published to feeder')

                    logger.info(f"pv names {self.pv_info.names}")
                    logger.info(f"pv p vals {self.p_pv_opti}")
                    logger.info(f"pv q vals {self.q_pv_opti}")

                    self.pub_opf_tap_values.publish(
                        LabelledArray(array=list(self.xmer_value_opti), unique_ids=self.tap_info.names).json())
                    logger.info(f'pub_tap_values published to feeder')


            granted_time = h.helicsFederateRequestTime(self.opf_fed, h.HELICS_TIME_MAXTIME)
            # logger.info(f'granted time in the for loop --> {granted_time}')

        self.destroy()

    def destroy(self):
        "Finalize and destroy the federates"
        h.helicsFederateDisconnect(self.opf_fed)
        print("Federate disconnected")

        h.helicsFederateFree(self.opf_fed)
        h.helicsCloseLibrary()


        
if __name__ == '__main__':
    logger.info(f'in opf federate.py')

    with open("opf_static_inputs.json") as f:
        config = json.load(f)
        # federate_name = config["name"]

    with open("opf_input_mapping.json") as f:
        input_mapping = json.load(f)

    opf_fed = OptimalPowerFlowFederate(config, input_mapping)
    opf_fed.go_opf()