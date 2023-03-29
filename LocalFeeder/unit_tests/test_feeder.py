import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import sender_cosim
import FeederSimulator
import dss_functions
import numpy as np
import pytest
import logging
import pandas as pd
import opendssdirect as dss

@pytest.fixture()
def federate_config():
    return FeederSimulator.FeederConfig(**{
        "use_smartds": False,
        "profile_location": "gadal_ieee123/profiles",
        "opendss_location": "gadal_ieee123/qsts",
        "sensor_location": "gadal_ieee123/sensors.json",
        "start_date": "2017-01-01 00:00:00",
        "number_of_timesteps": 96,
        "run_freq_sec": 900,
        "topology_output": "../../outputs/topology.json",
        "name": "feeder"
    })


def test_y_matrices(federate_config):
    logging.info("Loading sim")
    sim = sender_cosim.setup_sim(federate_config)
    Y = sim.get_y_matrix()
    directly = dss_functions.get_y_matrix_directly(dss)

    error_directly = np.max(np.abs(directly - Y))
    logging.info(f"Error directly: {error_directly}")
    assert error_directly < 0.1, "Directly is far away"

    directly_calcv = dss_functions.get_y_matrix_calcv(dss)
    error_calcv = np.max(np.abs(directly_calcv - Y))
    logging.info(f"Error calcv: {error_calcv}")
    assert error_calcv < 0.1

    dss_functions.snapshot_run(dss)

    Y2 = sim.get_y_matrix()
    error_Y2 = np.max(np.abs(Y2 - Y))
    logging.info(f"Error calcv: {error_Y2}")
    assert error_Y2 < 0.1

    dss_functions.snapshot_run(dss)

    directly2 = dss_functions.get_y_matrix_directly(dss)
    error_directly2 = np.max(np.abs(directly2 - Y))
    logging.info(f"Error directly2: {error_directly2}")
    assert error_directly2 < 0.1

    dss_functions.snapshot_run(dss)

    directly_calcv2 = dss_functions.get_y_matrix_calcv(dss)
    error_calcv2 = np.max(np.abs(directly_calcv2 - Y))
    logging.info(f"Error calcv 2: {error_calcv2}")
    assert error_calcv2 > 0.1

    sim.solve(0,0)

    Y2 = sim.get_y_matrix()
    error_Y2 = np.max(np.abs(Y2 - Y))
    logging.info(f"Error calcv: {error_Y2}")
    assert error_Y2 < 0.1

    sim.solve(0,0)

    directly2 = dss_functions.get_y_matrix_directly(dss)
    error_directly2 = np.max(np.abs(directly2 - Y))
    logging.info(f"Error directly2: {error_directly2}")
    assert error_directly2 < 0.1

    sim.solve(0,0)

    directly_calcv2 = dss_functions.get_y_matrix_calcv(dss)
    error_calcv2 = np.max(np.abs(directly_calcv2 - Y))
    logging.info(f"Error calcv 2: {error_calcv2}")
    assert error_calcv2 < 0.1


def test_simulation(federate_config):
    logging.info("Loading sim")
    sim = sender_cosim.setup_sim(federate_config)
    startup(sim)
    Y = sim.get_y_matrix()
    getting_and_concatentating_data(sim)
    initial_data(sim, federate_config)
    simulation_middle(sim, Y)


def startup(sim):
    assert sim._feeder_file is not None
    assert sim._AllNodeNames is not None
    assert sim._circuit is not None
    assert sim._AllNodeNames is not None


def getting_and_concatentating_data(sim):
    PQ_load = sim.get_PQs_load(static=True)
    PQ_PV = sim.get_PQs_pv(static=True)
    PQ_gen = sim.get_PQs_gen(static=True)
    PQ_cap = sim.get_PQs_cap(static=True)

    n_nodes = len(PQ_load)
    pv_real, pv_imag = sender_cosim.xarray_to_powers(PQ_PV, equipment_type=["PVSystem"]*n_nodes)
    gen_real, gen_imag = sender_cosim.xarray_to_powers(PQ_gen, equipment_type=["Generator"]*n_nodes)
    test_real = sender_cosim.concat_powers(pv_real, gen_real)
    assert test_real.values[5] == PQ_PV.data[5].real
    assert test_real.ids[5] == PQ_PV.bus.data[5]
    PQ_injections_all = PQ_load + PQ_PV + PQ_gen + PQ_cap
    assert np.all(PQ_injections_all.bus == PQ_load.bus)


def initial_data(sim, federate_config):
    initial_data = sender_cosim.get_initial_data(sim, federate_config)
    assert initial_data is not None


def simulation_middle(sim, Y):
    # this one may need to be properly ordered
    logging.info(f"Current directory : {os.getcwd()}")
    sim.solve(0,0)

    current_data = sender_cosim.get_current_data(sim, Y)
    df = pd.DataFrame({"pq": current_data.PQ_injections_all, "voltages": current_data.feeder_voltages})
    logging.info(df.describe())
    #assert np.max(current_data.PQ_injections_all) > 0.1
    assert np.max(current_data.feeder_voltages) > 50

    bad_bus_names = sender_cosim.where_power_unbalanced(current_data.PQ_injections_all, current_data.calculated_power)
    assert len(bad_bus_names) == 0
