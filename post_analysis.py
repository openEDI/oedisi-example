#!/usr/bin/env python3
import pyarrow.feather as feather
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

voltage_real = feather.read_feather("recorder_voltage_real/data.feather")
voltage_imag = feather.read_feather("recorder_voltage_imag/data.feather")

true_voltages = voltage_real.drop('time', axis=1) + 1j * voltage_imag.drop('time', axis=1)
time = voltage_real["time"]

voltage_mag = feather.read_feather("recorder_voltage_mag/data.feather")
voltage_angle = feather.read_feather("recorder_voltage_angle/data.feather")

estimated_voltages = voltage_mag.drop('time', axis=1) * np.exp(
    1j * voltage_angle
)

plt.plot(time, np.array(voltage_angle.drop('time', axis=1)))
plt.show()

true_angles = np.angle(true_voltages)
plt.plot(time, true_angles)
plt.show()
