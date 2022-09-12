#!/usr/bin/env python3
import pyarrow.feather as feather
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import json
import os
from gadal.gadal_types.data_types import MeasurementArray, AdmittanceMatrix, Topology

voltage_real = feather.read_feather(os.path.join("recorder_voltage_real","data.feather"))
voltage_imag = feather.read_feather(os.path.join("recorder_voltage_imag","data.feather"))

with open(os.path.join("local_feeder","topology.json")) as f:
    topology = Topology.parse_obj(json.load(f))
    base_voltages = np.array(topology.base_voltage_magnitudes.values)

voltage_mag = feather.read_feather(os.path.join("recorder_voltage_mag","data.feather"))
voltage_angle = feather.read_feather(os.path.join("recorder_voltage_angle","data.feather"))

estimated_voltages = voltage_mag.drop('time', axis=1) * np.exp(1j * voltage_angle.drop('time', axis=1))
time = voltage_mag["time"]

true_voltages = voltage_real.drop('time', axis=1) + 1j * voltage_imag.drop('time', axis=1)
true_times = voltage_real["time"]

def map_to_closest_time(time, true_times):
    closest_estimated_time_i = []
    prev_i = 0
    for t in time:
        i = prev_i
        while i < len(true_times) and true_times[i] <= t:
            i += 1
        if i != prev_i:
            closest_estimated_time_i.append(i - 1)
        else:
            break
    return closest_estimated_time_i

true_voltages = true_voltages.iloc[
    map_to_closest_time(time, true_times),:
]

def plots(true_voltages, estimated_voltages, time=0, unit="kV"):
    n_nodes = true_voltages.shape[0]
    x_axis = np.arange(n_nodes)
    fig1, ax = plt.subplots(figsize=(10,10))

    ax.bar(x_axis, np.angle(estimated_voltages))
    ax.bar(x_axis, np.angle(true_voltages), width=0.5)

    #ax.set_xticks(x_axis, true_voltages.index, rotation=-90, fontsize=5)
    #ax.set_tick_params(axis='x', labelsize=5, rotation=-90)
    ax.set_xlabel('Node number')
    ax.set_ylabel('Voltage Angles')
    ax.legend(['Estimated', 'True'])
    ax.set_title(f"Voltage Angles at t={time}")

    fig2, ax = plt.subplots(figsize=(10,10))
    ax.plot(x_axis, np.abs(estimated_voltages), '-o')
    ax.plot(x_axis, np.abs(true_voltages), '-o')
    #ax.set_xticks(x_axis, true_voltages.index, rotation=-90, fontsize=5)
    ax.set_xlabel('Node number')
    ax.set_ylabel(f'Voltage Magnitudes ({unit})')
    ax.legend(['Estimated', 'True'])
    ax.set_title(f"Voltage Magnitudes at t={time}")
    return fig1, fig2


def errors(true_voltages, estimated_voltages):
    true_mag = np.abs(true_voltages)
    nonzero_parts = true_mag != 0.0
    MAPE = np.mean(
        np.array(np.abs(true_mag - np.abs(estimated_voltages))
                / true_mag)[nonzero_parts]
        * 100
    )
    angle_difference = np.abs(np.angle(true_voltages) - np.angle(estimated_voltages))
    angle_difference[angle_difference >= np.pi] = 2*np.pi - angle_difference[angle_difference >= np.pi]
    MAE = np.mean(np.array(angle_difference)[nonzero_parts] * 180 / np.pi)
    return MAPE, MAE


def error_table(true_voltages, estimated_voltage):
    error_table = []
    for i, t in enumerate(time):
        MAPE, MAE = errors(true_voltages.iloc[i,:], estimated_voltages.iloc[i,:])
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
    return fig

err_table = error_table(true_voltages, estimated_voltages)
plot_errors(err_table).savefig("errors.png")
MAPE, MAE = errors(true_voltages, estimated_voltages)
print(f"MAPE = {MAPE}, MAE={MAE}")
fig1, fig2 = plots(true_voltages.iloc[0,:] / base_voltages, estimated_voltages.iloc[0,:] / base_voltages, unit="p.u.")
fig1.savefig("voltage_angles_0.png")
fig2.savefig("voltage_magnitudes_0.png")
if len(true_voltages) >=96:
    fig1, fig2 = plots(true_voltages.iloc[95,:] / base_voltages, estimated_voltages.iloc[95,:] / base_voltages, time=95, unit="p.u.")
    fig1.savefig("voltage_angles_95.png")
    fig2.savefig("voltage_magnitudes_95.png")

