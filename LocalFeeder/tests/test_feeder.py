import logging
import os
import sys

import numpy as np
import pandas as pd
import plotille
import pytest
import xarray as xr

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import FeederSimulator
import sender_cosim


@pytest.fixture()
def federate_config():
    return FeederSimulator.FeederConfig(
        **{
            "use_smartds": False,
            "profile_location": "gadal_ieee123/profiles",
            "opendss_location": "gadal_ieee123/qsts",
            "sensor_location": "gadal_ieee123/sensors.json",
            "start_date": "2017-01-01 00:00:00",
            "number_of_timesteps": 96,
            "run_freq_sec": 900,
            "topology_output": "../../outputs/topology.json",
            "name": "feeder",
        }
    )


def plot_y_matrix(Y):
    Y_max = np.max(np.abs(Y))

    width = Y.shape[0] // 2
    height = Y.shape[1] // 4
    cvs = plotille.Canvas(width, height, mode="rgb")
    arrayY = np.log(np.abs(Y.toarray())[: (4 * height), : (2 * width)] + 1)
    flatY = arrayY.reshape(arrayY.size)
    max_flatY = flatY.max()
    cvs.braille_image([int(255 * np.abs(y) / max_flatY) for y in flatY])
    print(f"Y max: {Y_max}")
    print(cvs.plot())


def test_ordering(federate_config):
    logging.info("Loading sim")
    sim = FeederSimulator.FeederSimulator(federate_config)
    Y = sim.get_y_matrix()

    with pytest.raises(AssertionError, match=".*DISABLED_RUN.*"):
        sim.solve(0, 0)
    with pytest.raises(AssertionError, match=".*DISABLED_RUN.*"):
        _ = sim.get_voltages_actual()

    sim.snapshot_run()
    base_voltages = sim.get_voltages_snapshot()
    assert np.max(base_voltages) > 100
    Y2 = sim.get_y_matrix()
    assert np.max(np.abs(Y - Y2)) < 0.001

    sim.snapshot_run()
    sim.solve(0, 0)
    feeder_voltages = sim.get_voltages_actual()
    assert np.max(feeder_voltages) > 100
    with pytest.raises(AssertionError, match=".*SOLVE_AT_TIME.*"):
        _ = sim.get_voltages_snapshot()


def rtheta_to_xy(r, theta):
    x = np.array(r * np.cos(theta))
    y = np.array(r * np.sin(theta))
    return x, y


def test_voltages(federate_config):
    logging.info("Loading sim")
    sim = FeederSimulator.FeederSimulator(federate_config)
    startup(sim)

    # Get Voltages
    base = sim.get_base_voltages()
    sim.disabled_run()
    sim.initial_disabled_solve()
    disabled_solve = sim.get_disabled_solve_voltages()
    sim.snapshot_run()
    snapshot = sim.get_voltages_snapshot()
    sim.solve(0, 0)
    actuals = sim.get_voltages_actual()

    # Plot magnitudes
    fig = plotille.Figure()
    fig.width = 60
    fig.height = 30
    # fig.set_x_limits(min_=-3, max_=3)
    fig.set_y_limits(min_=0, max_=2)
    fig.color_mode = "byte"
    fig.plot(
        range(len(base)), np.abs(disabled_solve / base).data, lc=50, label="Disabled"
    )
    fig.plot(range(len(base)), np.abs(snapshot / base).data, lc=75, label="Snapshot")
    fig.plot(range(len(base)), np.abs(actuals / base).data, lc=100, label="Actuals")
    print("\n" + fig.show(legend=True))

    # Plot magnitudes
    fig = plotille.Figure()
    fig.width = 60
    fig.height = 30
    # fig.set_x_limits(min_=-3, max_=3)
    fig.set_y_limits(min_=0.9, max_=1.1)
    fig.color_mode = "byte"
    fig.plot(
        range(len(base)), np.abs(disabled_solve / base).data, lc=50, label="Disabled"
    )
    fig.plot(range(len(base)), np.abs(snapshot / base).data, lc=75, label="Snapshot")
    fig.plot(range(len(base)), np.abs(actuals / base).data, lc=100, label="Actuals")
    print("\n" + fig.show(legend=True))

    # Plot angles better
    r = xr.DataArray(range(len(base)), {"bus": base.bus.data}) + 100
    fig = plotille.Figure()
    fig.width = 60
    fig.height = 30
    fig.set_x_limits(min_=-400, max_=400)
    fig.set_y_limits(min_=-400, max_=400)
    fig.color_mode = "byte"
    fig.scatter(*rtheta_to_xy(r, np.angle(base)), lc=50, label="Disabled")
    fig.scatter(*rtheta_to_xy(r, np.angle(snapshot)), lc=75, label="Snapshot")
    fig.scatter(*rtheta_to_xy(r, np.angle(actuals)), lc=100, label="Actuals")
    print("\n" + fig.show(legend=True))

    # Plot angles better
    r = xr.DataArray(range(len(base)), {"bus": base.bus.data}) + 100
    fig = plotille.Figure()
    fig.width = 90
    fig.height = 30
    fig.set_x_limits(min_=0, max_=400)
    fig.set_y_limits(min_=-50, max_=50)
    fig.color_mode = "byte"
    fig.scatter(*rtheta_to_xy(r, np.angle(base)), lc=50, label="Disabled")
    fig.scatter(*rtheta_to_xy(r, np.angle(snapshot)), lc=75, label="Snapshot")
    fig.scatter(*rtheta_to_xy(r, np.angle(actuals)), lc=100, label="Actuals")
    print("\n" + fig.show(legend=True))


def test_simulation(federate_config):
    logging.info("Loading sim")
    sim = FeederSimulator.FeederSimulator(federate_config)
    startup(sim)
    Y = sim.get_y_matrix()
    plot_y_matrix(Y)
    getting_and_concatentating_data(sim)
    initial_data(sim, federate_config)
    sim.snapshot_run()
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
    pv_real, pv_imag = sender_cosim.xarray_to_powers(
        PQ_PV, equipment_type=["PVSystem"] * n_nodes
    )
    gen_real, gen_imag = sender_cosim.xarray_to_powers(
        PQ_gen, equipment_type=["Generator"] * n_nodes
    )
    test_real = sender_cosim.concat_powers(pv_real, gen_real)
    assert test_real.values[5] == PQ_PV.data[5].real
    assert test_real.ids[5] == PQ_PV.bus.data[5]
    PQ_injections_all = PQ_load + PQ_PV + PQ_gen + PQ_cap
    assert np.all(PQ_injections_all.bus == PQ_load.bus)


def initial_data(sim, federate_config):
    initial_data = sender_cosim.get_initial_data(sim, federate_config)
    # Plot magnitudes
    fig = plotille.Figure()
    fig.width = 60
    fig.height = 30
    # fig.set_x_limits(min_=-3, max_=3)
    fig.set_y_limits(min_=0, max_=3000)
    fig.color_mode = "byte"
    fig.plot(
        range(len(initial_data.topology.base_voltage_magnitudes.ids)),
        np.sort(initial_data.topology.base_voltage_magnitudes.values),
        lc=100,
        label="Base",
    )
    print("\n" + fig.show(legend=True))

    # Plot angles better
    r = np.array(range(len(initial_data.topology.base_voltage_magnitudes.ids))) + 100
    fig = plotille.Figure()
    fig.width = 60
    fig.height = 30
    fig.set_x_limits(min_=-400, max_=400)
    fig.set_y_limits(min_=-400, max_=400)
    fig.color_mode = "byte"
    fig.scatter(
        *rtheta_to_xy(r, initial_data.topology.base_voltage_angles.values),
        lc=50,
        label="Base",
    )
    print("\n" + fig.show(legend=True))

    df = pd.DataFrame(
        {
            "voltages": initial_data.topology.base_voltage_magnitudes.values,
            "angles": initial_data.topology.base_voltage_angles.values,
        }
    )
    logging.info(df.describe())
    df = pd.DataFrame(
        {"injections": initial_data.topology.injections.power_real.values}
    )
    logging.info(df.describe())
    assert initial_data is not None


def plot_complex_array(data, label="Voltages"):
    # Plot magnitudes
    fig = plotille.Figure()
    fig.width = 60
    fig.height = 30
    # fig.set_x_limits(min_=-3, max_=3)
    fig.set_y_limits(min_=0, max_=np.max(np.abs(data)) * 1.1)
    fig.color_mode = "byte"
    fig.plot(range(len(data)), np.abs(data), lc=100, label=label)
    print("\n" + fig.show(legend=True))

    # Plot angles better
    r = np.array(range(len(data))) + 100
    fig = plotille.Figure()
    fig.width = 60
    fig.height = 30
    fig.set_x_limits(min_=-400, max_=400)
    fig.set_y_limits(min_=-400, max_=400)
    fig.color_mode = "byte"
    fig.scatter(*rtheta_to_xy(r, np.angle(data)), lc=50, label=label)
    print("\n" + fig.show(legend=True))


def simulation_middle(sim, Y):
    # this one may need to be properly ordered
    logging.info(f"Current directory : {os.getcwd()}")
    sim.solve(0, 0)

    current_data = sender_cosim.get_current_data(sim, Y)
    df = pd.DataFrame(
        {
            "pq": current_data.PQ_injections_all,
            "voltages": np.abs(current_data.feeder_voltages),
            "phases": np.angle(current_data.feeder_voltages),
        }
    )
    logging.info(df.describe())
    print("Feeder Voltages")
    plot_complex_array(current_data.feeder_voltages.data, label="Feeder Voltages")
    assert np.max(current_data.feeder_voltages) > 50

    plot_complex_array(current_data.PQ_injections_all.data, label="PQ injection")

    diff = np.abs(
        current_data.PQ_injections_all - (-current_data.calculated_power)
    ) * np.exp(
        1j
        * (
            np.angle(current_data.PQ_injections_all)
            - np.angle(-current_data.calculated_power)
        )
    )

    plot_complex_array(diff.data, label="Calculated - Injected")

    bad_bus_names = sender_cosim.where_power_unbalanced(
        current_data.PQ_injections_all, current_data.calculated_power
    )
    assert len(bad_bus_names) == 0
