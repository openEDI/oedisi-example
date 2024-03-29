#!/usr/bin/env python3
import pyarrow.feather as feather
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import json
import os
from oedisi.types.data_types import MeasurementArray, AdmittanceMatrix, Topology
import argparse


parser = argparse.ArgumentParser(description="Create plots")
parser.add_argument(
    "directory",
    nargs="?",
    default="outputs",
    help="directory to look for voltage_measurements",
)
args = parser.parse_args()

voltage_real = feather.read_feather(
    os.path.join(args.directory, "voltage_real.feather")
)
voltage_imag = feather.read_feather(
    os.path.join(args.directory, "voltage_imag.feather")
)

with open(os.path.join(args.directory, "topology.json")) as f:
    topology = Topology.parse_obj(json.load(f))
    base_voltage_df = pd.DataFrame(
        {
            "id": topology.base_voltage_magnitudes.ids,
            "value": topology.base_voltage_magnitudes.values,
        }
    )
    base_voltage_df.set_index("id", inplace=True)
    base_voltages = base_voltage_df["value"]

true_voltages = voltage_real.drop("time", axis=1) + 1j * voltage_imag.drop(
    "time", axis=1
)
true_voltages["time"] = voltage_real["time"]
true_voltages.set_index("time", inplace=True)

voltage_mag = feather.read_feather(os.path.join(args.directory, "voltage_mag.feather"))
estimated_time = voltage_mag["time"]
voltage_mag.drop("time", axis=1)
voltage_angle = feather.read_feather(
    os.path.join(args.directory, "voltage_angle.feather")
).drop("time", axis=1)

estimated_voltages = voltage_mag * np.exp(1j * voltage_angle)
estimated_voltages["time"] = estimated_time
estimated_voltages.set_index("time", inplace=True)

time_intersection = pd.merge(
    true_voltages, estimated_voltages, left_index=True, right_index=True
).index.to_numpy()

estimated_voltages = estimated_voltages.loc[time_intersection, :]
true_voltages = true_voltages.loc[time_intersection, :]

estimated_voltages = estimated_voltages.reindex(true_voltages.columns, axis=1)


def plots(true_voltages, estimated_voltages, time=0, unit="kV"):
    n_nodes = true_voltages.shape[0]
    x_axis = np.arange(n_nodes)
    fig1, ax = plt.subplots(figsize=(10, 10))

    ax.bar(x_axis, np.angle(estimated_voltages))
    ax.bar(x_axis, np.angle(true_voltages), width=0.5)

    # ax.set_xticks(x_axis, true_voltages.index, rotation=-90, fontsize=5)
    # ax.set_tick_params(axis='x', labelsize=5, rotation=-90)
    ax.set_xlabel("Node number")
    ax.set_ylabel("Voltage Angles")
    ax.legend(["Estimated", "True"])
    ax.set_title(f"Voltage Angles at t={time}")

    fig2, ax = plt.subplots(figsize=(10, 10))
    ax.plot(x_axis, np.abs(estimated_voltages), "-o")
    ax.plot(x_axis, np.abs(true_voltages), "-o")
    # ax.set_xticks(x_axis, true_voltages.index, rotation=-90, fontsize=5)
    ax.set_xlabel("Node number")
    ax.set_ylabel(f"Voltage Magnitudes ({unit})")
    ax.legend(["Estimated", "True"])
    ax.set_title(f"Voltage Magnitudes at t={time}")
    return fig1, fig2


def errors(true_voltages, estimated_voltages):
    true_mag = np.abs(true_voltages)
    nonzero_parts = true_mag != 0.0
    MAPE = np.mean(
        np.array(np.abs(true_mag - np.abs(estimated_voltages)) / true_mag)[
            nonzero_parts
        ]
        * 100
    )
    angle_difference = np.abs(np.angle(true_voltages) - np.angle(estimated_voltages))
    angle_difference[angle_difference >= np.pi] = (
        2 * np.pi - angle_difference[angle_difference >= np.pi]
    )
    MAE = np.mean(np.array(angle_difference)[nonzero_parts] * 180 / np.pi)
    return MAPE, MAE


def error_table(true_voltages, estimated_voltages):
    error_table = []
    for i, t in enumerate(true_voltages.index):
        MAPE, MAE = errors(true_voltages.iloc[i, :], estimated_voltages.iloc[i, :])
        error_table.append({"t": t, "MAPE": MAPE, "MAE": MAE})
    return pd.DataFrame(error_table)


def plot_errors(err_table):
    fig, ax = plt.subplots()
    ax.plot(err_table["t"], err_table["MAPE"])
    ax.plot(err_table["t"], err_table["MAE"])
    ax.legend(["MAPE (magnitudes)", "MAE (angles)"])
    ax.set_ylabel("Error")
    ax.set_xlabel("Time (15 minute)")
    ax.set_title("Voltage Errors")
    ax.set_xticks(err_table["t"][::5], err_table["t"][::5], rotation=-25, fontsize=5)
    return fig


err_table = error_table(true_voltages, estimated_voltages)
plot_errors(err_table).savefig("errors.png")
MAPE, MAE = errors(true_voltages, estimated_voltages)
print(f"MAPE = {MAPE}, MAE={MAE}")
fig1, fig2 = plots(
    true_voltages.iloc[0, :] / base_voltages,
    estimated_voltages.iloc[0, :] / base_voltages,
    unit="p.u.",
)
fig1.savefig("voltage_angles_0.png")
fig2.savefig("voltage_magnitudes_0.png")
if len(true_voltages) >= 94:
    fig1, fig2 = plots(
        true_voltages.iloc[93, :] / base_voltages,
        estimated_voltages.iloc[93, :] / base_voltages,
        time=93,
        unit="p.u.",
    )
    fig1.savefig("voltage_angles_95.png")
    fig2.savefig("voltage_magnitudes_95.png")
