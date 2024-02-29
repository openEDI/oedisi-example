import plotille
import pandas as pd
import numpy as np
from oedisi.types.data_types import Topology
import click
from os.path import join


def extract_data(path):
    reference_power_real = pd.read_feather(join(path, "reference_power_real.feather"))
    reference_power_imag = pd.read_feather(join(path, "reference_power_imag.feather"))
    power_real = pd.read_feather(join(path, "power_real.feather"))
    power_imag = pd.read_feather(join(path, "power_imag.feather"))
    reference_voltage_real = pd.read_feather(
        join(path, "reference_voltage_real.feather")
    )
    reference_voltage_imag = pd.read_feather(
        join(path, "reference_voltage_imag.feather")
    )
    voltage_real = pd.read_feather(join(path, "voltage_real.feather"))
    voltage_imag = pd.read_feather(join(path, "voltage_imag.feather"))

    topology = Topology.parse_file(join(path, "topology.json"))

    base_voltage_magnitudes = np.array(topology.base_voltage_magnitudes.values)

    reference_voltage = reference_voltage_real.drop(
        "time", axis=1
    ) + 1j * reference_voltage_imag.drop("time", axis=1)
    voltage = voltage_real.drop("time", axis=1) + 1j * voltage_imag.drop("time", axis=1)

    # Repeated for reference power
    reference_power = reference_power_real.drop(
        "time", axis=1
    ) + 1j * reference_power_imag.drop("time", axis=1)
    power = power_real.drop("time", axis=1) + 1j * power_imag.drop("time", axis=1)

    reference_time = reference_voltage_real.time
    time = voltage_real.time
    return (
        time,
        voltage,
        power,
        reference_time,
        reference_voltage,
        reference_power,
        base_voltage_magnitudes,
    )


@click.command(help="Run opf analysis in directory")
@click.argument(
    "path",
    default="outputs",
    type=click.Path(),
)
def run_analysis(path):
    (
        time,
        voltage,
        power,
        reference_time,
        reference_voltage,
        reference_power,
        base_voltage_magnitudes,
    ) = extract_data(path)
    num_time, num_nodes = reference_voltage.shape

    voltage_magnitude = (
        np.abs(voltage[time.isin(reference_time)].reset_index(drop=True))
        / base_voltage_magnitudes
    )
    reference_voltage_magnitude = np.abs(reference_voltage) / base_voltage_magnitudes
    magnitude_difference = voltage_magnitude - reference_voltage_magnitude

    fig = plotille.Figure()
    fig.width = 100
    fig.height = 70
    fig.plot(np.arange(num_nodes), magnitude_difference.iloc[40, :])
    print(fig.show())

    fig = plotille.Figure()
    fig.width = 100
    fig.height = 70
    fig.plot(np.arange(len(reference_time)), magnitude_difference.max(axis=1))
    print(fig.show())

    for i in range(30, 70, 10):
        fig = plotille.Figure()
        fig.width = 100
        fig.height = 70
        fig.plot(
            np.arange(num_nodes),
            reference_voltage_magnitude.iloc[i, :],
            label="Reference Voltage",
        )
        fig.plot(np.arange(num_nodes), voltage_magnitude.iloc[i, :], label="Voltage")
        fig.plot([0, num_nodes - 1], [0.95, 0.95])
        fig.plot([0, num_nodes - 1], [1.05, 1.05])
        print()
        print("Voltage at Time:", time[i])
        print(fig.show(legend=True))
    # (np.abs(voltage[time.isin(reference_time)].reset_index(drop=True)) - np.abs(reference_voltage)) / base_voltage_magnitudes

    power_magnitude_difference = np.abs(reference_power) - np.abs(
        power[time.isin(reference_time)].reset_index(drop=True)
    )
    for i in range(30, 70, 10):
        fig = plotille.Figure()
        fig.width = 100
        fig.height = 70
        fig.plot(np.arange(num_nodes), power_magnitude_difference.iloc[i, :])
        print()
        print("Power at Time:", time[i])
        print(fig.show())


if __name__ == "__main__":
    run_analysis()
