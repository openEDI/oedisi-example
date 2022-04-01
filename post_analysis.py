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

n_nodes = true_voltages.shape[1]
x_axis = np.arange(n_nodes)
plt.bar(x_axis, np.array(voltage_angle.iloc[0,:]))

true_angles = np.angle(true_voltages)
plt.bar(x_axis, true_angles[0,:], width=0.5)

plt.xticks(x_axis, true_voltages.columns, rotation=-90)
plt.legend(['Estimated', 'True'])
plt.show()
