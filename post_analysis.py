#!/usr/bin/env python3
import pyarrow.feather as feather
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

voltage_real = feather.read_feather("recorder_voltage_real/data.feather")
voltage_imag = feather.read_feather("recorder_voltage_imag/data.feather")

true_voltages = voltage_real.drop('time', axis=1) + 1j * voltage_imag.drop('time', axis=1)
time = voltage_real["time"]

voltage_mag = feather.read_feather("recorder_voltage_mag/data.feather").drop('time', axis=1)
voltage_angle = feather.read_feather("recorder_voltage_angle/data.feather").drop('time', axis=1)

estimated_voltages = voltage_mag * np.exp(1j * voltage_angle)

def plots(true_voltages, estimated_voltages):
    n_nodes = true_voltages.shape[1]
    x_axis = np.arange(n_nodes)
    plt.bar(x_axis, np.angle(estimated_voltages)[0,:])

    plt.bar(x_axis, np.angle(true_angles)[0,:], width=0.5)

    plt.xticks(x_axis, true_voltages.columns, rotation=-90)
    plt.ylabel('Voltage Angles')
    plt.legend(['Estimated', 'True'])
    plt.show()

    plt.bar(x_axis, np.abs(estimated_voltages).iloc[0,:])
    plt.bar(x_axis, np.abs(true_voltages).iloc[0,:], width=0.5)
    plt.xticks(x_axis, true_voltages.columns, rotation=-90)
    plt.ylabel('Voltage Magnitudes')
    plt.legend(['Estimated', 'True'])
    plt.show()

def errors(true_voltages, estimated_voltages):
    true_mag = np.abs(true_voltages)
    MAPE = np.mean(
        np.array(np.abs(true_mag - np.abs(estimated_voltages))
                / true_mag)
        * 100
    )
    MAE = np.mean(
        np.array(np.abs(
            np.angle(true_voltages) - np.angle(estimated_voltages)
        )) % (2*np.pi) * 180 / np.pi
    )
    print(f"MAPE = {MAPE}, MAE={MAE}")


errors(true_voltages, estimated_voltages)
