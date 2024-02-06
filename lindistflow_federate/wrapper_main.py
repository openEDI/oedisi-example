# -*- coding: utf-8 -*-
"""
Created on Sun March 10 12:58:46 2023
@author: poud579 & Rabayet
"""

import os
from adms_agent_V2 import dist_OPF
import networkx as nx
import numpy as np
import pandas as pd
import copy
import json
import time
import matplotlib.pyplot as plt
import cvxpy as cp
# import timeit

# This is a test update for the git
global mult, mult_pv, mult_load, mult_sec_pv, primary_kv_level


def area_info(G, edge, branch_sw_data, bus_info, sourcebus, area_source_bus):
    # Find area between the switches
    for e in edge:
        G.remove_edge(e[0], e[1])

    # Find area specific information
    # T = list(nx.bfs_tree(G, source = sourcebus).edges())
    # print("\n Number of Buses:", G.number_of_nodes(), "\n", "Number of Edges:", G.number_of_edges())
    # print('\n The number of edges in a Spanning tree is:', len(T))
    # print(list(nx.connected_components(G)))

    # List of sub-graphs. The one that has no sourcebus is the disconnected one
    sp_graph = list(nx.connected_components(G))
    for k in sp_graph:
        if sourcebus == area_source_bus:
            if sourcebus in k:
                area = k
                break
        else:
            if sourcebus not in k:
                area = k
                break

    bus_info_area_i = {}
    idx = 0
    sump = 0
    sumq = 0
    sumpv = 0
    for key, val_bus in bus_info.items():
        if key in area:
            if bus_info[key]['kv'] > primary_kv_level:
                bus_info_area_i[key] = {}
                bus_info_area_i[key]['idx'] = idx
                bus_info_area_i[key]['phases'] = bus_info[key]['phases']
                bus_info_area_i[key]['nodes'] = bus_info[key]['nodes']
                bus_info_area_i[key]['kv'] = bus_info[key]['kv']
                bus_info_area_i[key]['s_rated'] = (
                            bus_info[key]['pv'][0][0] + bus_info[key]['pv'][1][0] + bus_info[key]['pv'][2][0])
                bus_info_area_i[key]['pv'] = [[pv[0] * mult_pv, pv[1] * mult_pv] for pv in bus_info[key]['pv']]
                bus_info_area_i[key]['pq'] = [[pq[0] * mult_load, pq[1] * mult_load] for pq in bus_info[key]['pq']]
                sump += bus_info_area_i[key]['pq'][0][0]
                sump += bus_info_area_i[key]['pq'][1][0]
                sump += bus_info_area_i[key]['pq'][2][0]
                sumq += bus_info_area_i[key]['pq'][0][1]
                sumq += bus_info_area_i[key]['pq'][1][1]
                sumq += bus_info_area_i[key]['pq'][2][1]
                sumpv += bus_info_area_i[key]['pv'][0][0]
                sumpv += bus_info_area_i[key]['pv'][1][0]
                sumpv += bus_info_area_i[key]['pv'][2][0]
                idx += 1

    for key, val_bus in bus_info.items():
        if key in area:
            if bus_info[key]['kv'] < primary_kv_level:
                bus_info_area_i[key] = {}
                bus_info_area_i[key]['idx'] = idx
                bus_info_area_i[key]['phases'] = bus_info[key]['phases']
                bus_info_area_i[key]['nodes'] = bus_info[key]['nodes']
                bus_info_area_i[key]['kv'] = bus_info[key]['kv']
                # bus_info_area_i[key]['pv'] = bus_info[key]['pv']
                bus_info_area_i[key]['pv'] = [i * mult_sec_pv for i in bus_info[key]['pv']]
                bus_info_area_i[key]['pq'] = [i * mult_load for i in bus_info[key]['pq']]
                bus_info_area_i[key]['s_rated'] = (bus_info[key]['pv'][0])
                sump += bus_info_area_i[key]['pq'][0]
                sumq += bus_info_area_i[key]['pq'][1]
                sumpv += bus_info_area_i[key]['pv'][0]
                idx += 1
    idx = 0
    print(sump, sumq, sumpv)

    secondary_model = ['SPLIT_PHASE', 'TPX_LINE']
    branch_sw_data_area_i = {}
    nor_open = ['sw7', 'sw8']
    for key, val_bus in branch_sw_data.items():
        if val_bus['fr_bus'] in bus_info_area_i and val_bus['to_bus'] in bus_info_area_i:
            if branch_sw_data[key]['type'] not in secondary_model and key not in nor_open:
                branch_sw_data_area_i[key] = {}
                branch_sw_data_area_i[key]['idx'] = idx
                branch_sw_data_area_i[key]['type'] = branch_sw_data[key]['type']
                branch_sw_data_area_i[key]['from'] = bus_info_area_i[branch_sw_data[key]['fr_bus']]['idx']
                branch_sw_data_area_i[key]['to'] = bus_info_area_i[branch_sw_data[key]['to_bus']]['idx']
                branch_sw_data_area_i[key]['fr_bus'] = branch_sw_data[key]['fr_bus']
                branch_sw_data_area_i[key]['to_bus'] = branch_sw_data[key]['to_bus']
                branch_sw_data_area_i[key]['phases'] = branch_sw_data[key]['phases']
                if branch_sw_data[key]['type'] == 'SPLIT_PHASE':
                    branch_sw_data_area_i[key]['impedance'] = branch_sw_data[key]['impedance']
                    branch_sw_data_area_i[key]['impedance1'] = branch_sw_data[key]['impedance1']
                else:
                    branch_sw_data_area_i[key]['zprim'] = branch_sw_data[key]['zprim']
                idx += 1
    idx = 0
    for key, val_bus in branch_sw_data.items():
        if val_bus['fr_bus'] in bus_info_area_i and val_bus['to_bus'] in bus_info_area_i:
            if branch_sw_data[key]['type'] in secondary_model:
                branch_sw_data_area_i[key] = {}
                branch_sw_data_area_i[key]['idx'] = idx
                branch_sw_data_area_i[key]['type'] = branch_sw_data[key]['type']
                branch_sw_data_area_i[key]['from'] = bus_info_area_i[branch_sw_data[key]['fr_bus']]['idx']
                branch_sw_data_area_i[key]['to'] = bus_info_area_i[branch_sw_data[key]['to_bus']]['idx']
                branch_sw_data_area_i[key]['fr_bus'] = branch_sw_data[key]['fr_bus']
                branch_sw_data_area_i[key]['to_bus'] = branch_sw_data[key]['to_bus']
                branch_sw_data_area_i[key]['phases'] = branch_sw_data[key]['phases']
                if branch_sw_data[key]['type'] == 'SPLIT_PHASE':
                    branch_sw_data_area_i[key]['impedance'] = branch_sw_data[key]['impedance']
                    branch_sw_data_area_i[key]['impedance1'] = branch_sw_data[key]['impedance1']
                else:
                    branch_sw_data_area_i[key]['impedance'] = branch_sw_data[key]['zprim']
                idx += 1
    return branch_sw_data_area_i, bus_info_area_i


def split_graph(bus_info, branch_sw_xfmr):
    G = nx.Graph()
    for b in branch_sw_xfmr:
        G.add_edge(branch_sw_xfmr[b]['fr_bus'], branch_sw_xfmr[b]['to_bus'])

    # Finding the switch delimited areas and give the area specific information to agents
    sourcebus = '150'
    # area_info_swt = {'area_cen': {}, 'area_1': {}, 'area_2': {}, 'area_3': {}, 'area_4': {}, 'area_5': {}}
    area_info_swt = {'area_cen': {}}

    v_source = [1.0475, 1.0475, 1.0475]

    # Run the centralized power flow to get the real-time operating voltage
    area_info_swt['area_cen']['edges'] = [['54', '94'], ['151', '300']]  # [['13', '152'], ['18', '135']]
    area_info_swt['area_cen']['source_bus'] = '150'
    area_info_swt['area_cen']['vsrc'] = v_source
    edge = area_info_swt['area_cen']['edges']
    area_source_bus = area_info_swt['area_cen']['source_bus']
    G_area = copy.deepcopy(G)
    branch_sw_data_area_cen, bus_info_area_cen = area_info(G_area, edge, branch_sw_xfmr, bus_info, sourcebus,
                                                           area_source_bus)

    areas_info = {'bus_info': {}, 'branch_info': {}}
    areas_info['bus_info']['area_cen'] = bus_info_area_cen
    areas_info['branch_info']['area_cen'] = branch_sw_data_area_cen

    return areas_info, area_info_swt
def voltage_plot(bus_info, bus_voltage_area_cen, plot_node_voltage=False):
    # Plotting the voltage:
    if plot_node_voltage:
        Va = []
        Vb = []
        Vc = []
        for b in bus_voltage_area_cen:
            if '1' in bus_info[b]['phases']:
                Va.append(bus_voltage_area_cen[b]['A'])
            if '2' in bus_info[b]['phases']:
                Vb.append(bus_voltage_area_cen[b]['B'])
            if '3' in bus_info[b]['phases']:
                Vc.append(bus_voltage_area_cen[b]['C'])

        fig, ax1 = plt.subplots(1)
        ax1.scatter(range(1, len(Va) + 1), Va, c='blue', edgecolor='blue')
        ax1.scatter(range(1, len(Vb) + 1), Vb, c='green', edgecolor='green')
        ax1.scatter(range(1, len(Vc) + 1), Vc, c='red', edgecolor='red')
        ax1.set_xlabel('Bus Indices')
        ax1.set_ylabel('Voltage (p.u.)')
        # ax1.plot(range(1, len(Va) + 1), np.ones(len(Va)) * 1.05, 'r--')
        # ax1.plot(range(1, len(Va) + 1), np.ones(len(Va)) * 0.95, 'r--')
        ax1.plot(np.ones(100) * 1.05, 'r--')
        ax1.plot(np.ones(100) * 0.95, 'r--')
        # ax1.set_xlim(1, len(Va) + 1)

        # ax1.set_xlabel('Bus Indices', fontsize=18)
        # ax1.set_ylabel('Voltage (p.u.)', fontsize=18)
        # ax1.plot(range(1, len(Va) + 1), np.ones(len(Va)) * 1.05, 'r--')
        # ax1.set_xlim(1, len(Va) + 1)
        # ax1.tick_params(axis='x', labelsize=18)
        # ax1.tick_params(axis='y', labelsize=18)


        plt.ylim([0.9, 1.1])
        plt.legend(['Phase-A', 'Phase-B', 'Phase-C'])
        plt.show()
def OPF_voltage_plot(bus_info, bus_voltage_area_cen, plot_node_voltage=False):
    # Plotting the voltage:
    if plot_node_voltage:
        Va = []
        Vb = []
        Vc = []
        for b in bus_voltage_area_cen:
            if '1' in bus_info[b]['phases']:
                Va.append(bus_voltage_area_cen[b]['A'])
            if '2' in bus_info[b]['phases']:
                Vb.append(bus_voltage_area_cen[b]['B'])
            if '3' in bus_info[b]['phases']:
                Vc.append(bus_voltage_area_cen[b]['C'])

        fig, ax1 = plt.subplots(1)
        # ax1.scatter(range(1, len(Va) + 1), Va, c='white', edgecolor='blue')
        # ax1.scatter(range(1, len(Vb) + 1), Vb, c='white', edgecolor='blue')
        # ax1.scatter(range(1, len(Vc) + 1), Vc, c='white', edgecolor='blue')

        i = 0
        j  =0
        for idx, v in enumerate(Va):
            if v > 1.05:
                if i == 0:
                    ax1.scatter(idx, v, c='red', edgecolor='white',label = 'Over V limit')
                    i += 1
                else:
                    ax1.scatter(idx, v, c='red', edgecolor='white')
            else:
                if j == 0:
                    ax1.scatter(idx, v, c='white', edgecolor='blue',label = 'Within V limit')
                    j +=1
                else:
                    ax1.scatter(idx, v, c='white', edgecolor='blue')
        for idx, v in enumerate(Vb):
            if v > 1.05:
                if i == 0:
                    ax1.scatter(idx, v, c='red', edgecolor='white', label='Over V limit')
                    i += 1
                else:
                    ax1.scatter(idx, v, c='red', edgecolor='white')
            else:
                if j == 0:
                    ax1.scatter(idx, v, c='white', edgecolor='blue', label='Within V limit')
                    j += 1
                else:
                    ax1.scatter(idx, v, c='white', edgecolor='blue')
        for idx, v in enumerate(Vc):
            if v > 1.05:
                if i == 0:
                    ax1.scatter(idx, v, c='red', edgecolor='white', label='Over V limit')
                    i += 1
                else:
                    ax1.scatter(idx, v, c='red', edgecolor='white')
            else:
                if j == 0:
                    ax1.scatter(idx, v, c='white', edgecolor='blue', label='Within V limit')
                    j += 1
                else:
                    ax1.scatter(idx, v, c='white', edgecolor='blue')

        ax1.set_xlabel('Bus Indices', fontsize=18)
        ax1.set_ylabel('Voltage (p.u.)', fontsize=18)
        ax1.plot(range(1, len(Va) + 1), np.ones(len(Va)) * 1.05, 'r--')
        ax1.set_xlim(1, len(Va) + 1)
        ax1.tick_params(axis='x', labelsize=18)
        ax1.tick_params(axis='y', labelsize=18)

        plt.ylim([1.01, 1.06])
        plt.legend(fontsize=15)
        plt.show()
def save_optimal_Pgeneration_dss_file(dss_file_name,Control_variables_dict,kw_converter,bus_info_cen):
    current_directory = os.getcwd()
    file_path = os.path.join(current_directory, "OpenDSS_validations\\OpenDSS_files\\" + dss_file_name)
    # Erase the previous data:
    with open(file_path, 'w') as file:
        file.write("")

    for key, val_bus in Control_variables_dict.items():
        new_dg_str_a = ""
        new_dg_str_b = ""
        new_dg_str_c = ""
        if '1' in bus_info_cen[key]['phases']:
            new_dg_str_a = "New Generator.DG_S" + key + "a Bus1 =" + key + ".1 Phases = 1 kV=2.4 kW=" + \
                           str(Control_variables_dict[key][
                                   'A'] * kw_converter) + " kvar=0 model=1 Vmaxpu=2.0 Vminpu=0.1"

        if '2' in bus_info_cen[key]['phases']:
            new_dg_str_b = "New Generator.DG_S" + key + "b Bus1 =" + key + ".2 Phases = 1 kV=2.4 kW=" + \
                           str(Control_variables_dict[key][
                                   'B'] * kw_converter) + " kvar=0 model=1 Vmaxpu=2.0 Vminpu=0.1"

        if '3' in bus_info_cen[key]['phases']:
            new_dg_str_c = "New Generator.DG_S" + key + "c Bus1 =" + key + ".3 Phases = 1 kV=2.4 kW=" \
                           + str(
                Control_variables_dict[key]['C'] * kw_converter) + " kvar=0 model=1 Vmaxpu=2.0 Vminpu=0.1"

        # Open the file in append mode
        with open(file_path, 'a') as file:
            if not len(new_dg_str_a) == 0:
                file.write(new_dg_str_a + "\n")
            if not len(new_dg_str_b) == 0:
                file.write(new_dg_str_b + "\n")
            if not len(new_dg_str_c) == 0:
                file.write(new_dg_str_c + "\n")

    print("Optimal P generation DSS file is saved in  " + file_path)
def save_optimal_Qgeneration_dss_file(dss_file_name,Control_variables_dict,kw_converter,bus_info_cen):
    current_directory = os.getcwd()
    file_path = os.path.join(current_directory, "OpenDSS_validations\\OpenDSS_files\\" + dss_file_name)
    # Erase the previous data:
    with open(file_path, 'w') as file:
        file.write("")

    for key, val_bus in Control_variables_dict.items():
        new_dg_str_a = ""
        new_dg_str_b = ""
        new_dg_str_c = ""
        if '1' in bus_info_cen[key]['phases']:
            new_dg_str_a = "New Generator.DG_S" + key + "a Bus1 =" + key + ".1 Phases = 1 kV=2.4 kW=" + \
                           str(bus_info_cen[key]['pv'][0][0]/1000) + " kvar="+str(Control_variables_dict[key][
                                   'A'] * kw_converter) + " model=1 Vmaxpu=2.0 Vminpu=0.1"

        if '2' in bus_info_cen[key]['phases']:
            new_dg_str_b = "New Generator.DG_S" + key + "b Bus1 =" + key + ".2 Phases = 1 kV=2.4 kW=" + \
                           str(bus_info_cen[key]['pv'][1][0]/1000) + " kvar="+str(Control_variables_dict[key][
                                   'B'] * kw_converter) + " model=1 Vmaxpu=2.0 Vminpu=0.1"

        if '3' in bus_info_cen[key]['phases']:
            new_dg_str_c = "New Generator.DG_S" + key + "c Bus1 =" + key + ".3 Phases = 1 kV=2.4 kW=" + \
                           str(bus_info_cen[key]['pv'][2][0]/1000) + " kvar="+str(Control_variables_dict[key][
                                   'C'] * kw_converter) + " model=1 Vmaxpu=2.0 Vminpu=0.1"

        # Open the file in append mode
        with open(file_path, 'a') as file:
            if not len(new_dg_str_a) == 0:
                file.write(new_dg_str_a + "\n")
            if not len(new_dg_str_b) == 0:
                file.write(new_dg_str_b + "\n")
            if not len(new_dg_str_c) == 0:
                file.write(new_dg_str_c + "\n")

    print("Optimal Q generation DSS file is saved in  " + file_path)

def check_network_radiality(bus_info_cen,bus_info):
    # if not len(bus_info_cen)-len(branch_info_cen) == 1:
    if not len(bus_info_cen) - len(bus_info) == 0:
        print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
        print("                !!!!  ERROR  !!!!                   ")
        print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
        print("Stopping simulation prematurely")
        print("------------------------------------------------------------")
        print(" ->>> Network is NOT Radial. PLease check the network data")
        print("------------------------------------------------------------")
        exit()


def check_network_PowerFlow(branch_info_cen,bus_info_cen,agent_source_bus,agent_source_bus_idx,vsrc):
    # Power Flow Check:
    print('checking power flow')
    pf_flag = 1
    P_control = 1
    Q_control = 0

    solver_name = cp.ECOS
    print_LineFlows_Voltage = 0

    bus_voltage_area_cen, flow_area_cen, Control_variables_dict, kw_converter = dist_OPF(branch_info_cen, bus_info_cen,
                                                                                         agent_source_bus,
                                                                                         agent_source_bus_idx, vsrc,
                                                                                         pf_flag,
                                                                                         solver_name, P_control, Q_control,
                                                                                         print_LineFlows_Voltage, print_result=False)

    maxV = 0
    minV = 5
    for key, val in bus_voltage_area_cen.items():
        node_voltA = bus_voltage_area_cen[key]['A']
        node_voltB = bus_voltage_area_cen[key]['B']
        node_voltC = bus_voltage_area_cen[key]['C']
        node_max = max(node_voltA, node_voltB, node_voltC)
        node_min = min(node_voltA, node_voltB, node_voltC)
        if node_max > maxV:
            maxV = node_max
        if node_min < minV:
            minV = node_min

    if maxV > 1.2 or minV < 0.8:
        print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
        print("                !!!!  ERROR  !!!!                   ")
        print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
        print("Stopping simulation prematurely")
        print("----------------------------------------------------------------------------")
        print(" ->>> Network is not power flow feasible. Update the network parameter")
        print("----------------------------------------------------------------------------")
        exit()


def _save_node_voltage(bus_info, bus_voltage_area_cen,file_name):

    Va = []
    Vb = []
    Vc = []
    for b in bus_voltage_area_cen:
        if '1' in bus_info[b]['phases']:
            Va.append(bus_voltage_area_cen[b]['A'])
        if '2' in bus_info[b]['phases']:
            Vb.append(bus_voltage_area_cen[b]['B'])
        if '3' in bus_info[b]['phases']:
            Vc.append(bus_voltage_area_cen[b]['C'])

    data = {'Va': Va,
            'Vb': Vb,
            'Vc': Vc}

    current_directory = os.getcwd()
    file_path = os.path.join(current_directory, "solutions\\" + file_name)
    with open(file_path, 'w') as json_file:
        json.dump(data, json_file)

    print("Node voltages are saved in solutions\\"+ file_name )


def _save_control_variables(bus_info, Control_variables_dict,file_name):
    U = {}
    for b in Control_variables_dict:
        if '1' in bus_info[b]['phases']:
            if b not in U:
                U[b] = {}
            U[b][0] = Control_variables_dict[b]['A']
        if '2' in bus_info[b]['phases']:
            if b not in U:
                U[b] = {}
            U[b][1] = Control_variables_dict[b]['B']
        if '3' in bus_info[b]['phases']:
            if b not in U:
                U[b] = {}
            U[b][2] = Control_variables_dict[b]['C']

    current_directory = os.getcwd()
    file_path = os.path.join(current_directory, "solutions\\" + file_name)
    with open(file_path, 'w') as json_file:
        json.dump(U, json_file)

    print("Control Variables are saved in solutions\\"+ file_name )

if __name__ == '__main__':
    global mult, mult_pv, mult_load, primary_KV_level

    primary_KV_level = 0.4

    # load_mult = pd.read_csv('loadshape_15min.csv',
    #                         header=None)
    # pv_mult = pd.read_csv('PVshape_15min.csv',
    #                       header=None)


    # Load branch and bus info
    # bus_file = "bus_info_test_error.json"
    # branch_file = "branch_info_test_error.json"



    bus_file = "ckts//bus_info_test.json"
    branch_file = "ckts//branch_info_test.json"

    f = open(bus_file)
    bus_info = json.load(f)

    f = open(branch_file)
    branch_sw_xfmr = json.load(f)

    # Start the time series simulation
    bus_voltages = {}
    for t in range(52, 96):
        # print("Running time...................................... ", t)
        # mult_pv = pv_mult.iloc[:, 0][t]
        # mult_load = load_mult.iloc[:, 0][t]

        primary_kv_level = 0.4
        mult_load = 0.3
        mult_pv = 0.9
        mult_sec_pv = mult_pv
        print("t:", t,"; mult_pv:",mult_pv ,"; mult_load:",mult_load ,"; mult_sec_pv:",mult_sec_pv)

        areas_info, area_info_swt = split_graph(bus_info, branch_sw_xfmr)

        # Run centralized optimization
        agent_source_bus = area_info_swt['area_cen']['source_bus']
        bus_info_cen = areas_info['bus_info']['area_cen']
        branch_info_cen = areas_info['branch_info']['area_cen']
        agent_source_bus_idx = bus_info_cen[agent_source_bus]['idx']
        vsrc = area_info_swt['area_cen']['vsrc']

        print("Network error check starts")
        check_network_radiality(bus_info_cen,bus_info)

        check_network_PowerFlow(branch_info_cen,bus_info_cen,agent_source_bus,agent_source_bus_idx,vsrc)

        print("Power Flow result available without optimization")
        print("Central_OPF start")

        # OPF Starts:
        pf_flag = 0
        P_control = 1
        Q_control = 0

        solver_name = cp.ECOS
        print_LineFlows_Voltage = 0

        startTime = time.time()
        bus_voltage_area_cen, flow_area_cen, Control_variables_dict, kw_converter = dist_OPF(branch_info_cen, bus_info_cen,
                                                                          agent_source_bus, agent_source_bus_idx, vsrc, pf_flag,
                                                                          solver_name, P_control, Q_control,
                                                                          print_LineFlows_Voltage)

        ExecTime = (time.time() - startTime)

        if P_control == 1 and Q_control == 0:
            save_optimal_Pgeneration_dss_file("OPF_pvs.dss", Control_variables_dict, kw_converter, bus_info_cen)
        if P_control == 0 and Q_control == 1:
            save_optimal_Qgeneration_dss_file("OPF_pvs.dss", Control_variables_dict, kw_converter, bus_info_cen)

        # val_bus['pv'][0][0]
        print('Total Preparation and Execution Time: ' + str(ExecTime))
        print("---------------------------%%%%%--------------------------")

        _save_control_variables(bus_info, Control_variables_dict, file_name="control_variable.json")

        #Save Node voltages in a json file:
        _save_node_voltage(bus_info, bus_voltage_area_cen, file_name="voltage_opf.json")

        # # Plotting the voltage:
        voltage_plot(bus_info_cen, bus_voltage_area_cen, plot_node_voltage=0)
        OPF_voltage_plot(bus_info_cen, bus_voltage_area_cen, plot_node_voltage=1)

        exit()