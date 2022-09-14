import numpy as np
import matplotlib.pyplot as plt


opf_flag = False
print(f"reading data ")
node_data = np.genfromtxt('feeder/nodenames.csv',dtype=str, delimiter=' ')
voltage_data = np.genfromtxt('feeder/voltages.csv', delimiter=",")
active_power_load_data =np.genfromtxt('feeder/active_power_loads.csv', delimiter=",")
active_power_pv_data =np.genfromtxt('feeder/active_power_pv.csv',  delimiter=",")
reactive_power_pv_data =np.genfromtxt('feeder/reactive_power_pv.csv',  delimiter=",")
reactive_power_cap_data =np.genfromtxt('feeder/cap_reactive_power.csv',  delimiter=",")
print(f"all data read ")

taps_data =np.genfromtxt('feeder/taps.csv', delimiter=",")


node_len = len(node_data)
# intervals = voltage_data.shape[0]
# voltage_data_reshape = np.reshape(voltage_data, (intervals,node_len))
# active_power_load_data_reshape = np.reshape(active_power_load_data, (intervals,node_len))
# active_power_pv_data_reshape = np.reshape(active_power_pv_data, (intervals,node_len))
# reactive_power_pv_data_reshape = np.reshape(reactive_power_pv_data, (intervals,node_len))
# reactive_power_cap_data_reshape = np.reshape(reactive_power_cap_data, (intervals,node_len))
# tap_data_reshape = np.reshape(taps_data, (intervals,9))
node_data_new = []
for i in range(0, node_len):
    node_data_new.append(node_data[i].replace(',', ''))
node_data_new = np.array(node_data_new)

check = 1
load_idx = [97, 98, 99, 100, 101, 102, 145, 149]
pv_idx = [14, 38, 63, 88, 101, 121, 141, 154, 171, 185, 217, 241, 254]
cap_idx = [194, 195,196, 201, 205, 209]
voltage_idx = [194, 195,196, 201, 205, 209, 14, 38, 63, 88, 101, 121, 141, 154, 171, 185,  217, 241, 254]#, 194,195,196, 201, 205, 209]

if opf_flag is True:
    leg_str = 'with OPF'
else:
    leg_str = 'without OPF'
# voltage plots
#print(f"saving selected idx voltage plots")

fig, ax = plt.subplots()
ax.plot(voltage_data[:, voltage_idx])
ax.legend(node_data_new[voltage_idx])
ax.set_ylabel("Voltages (V)")
ax.set_xlabel("Time (seconds)")
ax.set_title("Voltage Magnitudes "+leg_str)
plt.savefig("Voltage Magnitudes Powers "+leg_str+'.png')
#print(f"saving voltage plots")
# pv p plots
print(f"saving selected idx pv active plots")

fig, ax = plt.subplots()
ax.plot(active_power_pv_data[:, pv_idx])
ax.legend(node_data_new[pv_idx])
ax.set_ylabel("Power (Watts)")
ax.set_xlabel("Time (seconds)")
ax.set_title("PV Active Powers "+leg_str)
plt.savefig("PV Active Powers "+leg_str+'.png')

# pv q plots
print(f"saving selected idx pv reactive plots")

fig, ax = plt.subplots()
ax.plot(reactive_power_pv_data[:, pv_idx])
ax.legend(node_data_new[pv_idx])
ax.set_ylabel("Power (Vars)")
ax.set_xlabel("Time (seconds)")
ax.set_title("PV Reactive Powers "+leg_str)
plt.savefig("PV Reactive Powers "+leg_str+'.png')

# cap q plots
print(f"saving selected idx cap banks plots")

fig, ax = plt.subplots()
ax.plot(reactive_power_cap_data[:, cap_idx])
ax.legend(node_data_new[cap_idx])
ax.set_ylabel("Power (Vars)")
ax.set_xlabel("Time (seconds)")
ax.set_title("Cap Reactive Powers "+leg_str)
plt.savefig("Cap Reactive Powers "+leg_str+'.png')

# load vals
print(f"saving selected idx load vals")

fig, ax = plt.subplots()
ax.plot(active_power_load_data[:, load_idx])
ax.legend(node_data_new[load_idx])
ax.set_ylabel("Load (Watts)")
ax.set_xlabel("Time (seconds)")
ax.set_title("Load Values "+leg_str)
plt.savefig("Load Values "+leg_str+'.png')