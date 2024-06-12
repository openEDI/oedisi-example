
"""
Created on Sun March 10 12:58:46 2023
@author: poud579 & Rabayet
"""

import networkx as nx
import logging

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.DEBUG)


def graph_process(branch_info: dict):
    graph = nx.Graph()
    edges = []
    for b in branch_info:
        if branch_info[b]['type'] != 'SWITCH':
            graph.add_edge(branch_info[b]['fr_bus'],
                           branch_info[b]['to_bus'])
        else:
            edges.append([branch_info[b]['fr_bus'],
                          branch_info[b]['to_bus']])
    return graph, edges


def area_info(branch_info: dict, bus_info: dict, source_bus: str):
    # System's base definition
    mult_pv = 1.0
    mult_sec_pv = 1.0
    mult_load = 1.0
    primary_kv_level = 0.12
    v_source = [1.0475, 1.0475, 1.0475]
    G, open_switches = graph_process(branch_info)

    area_info_swt = {'area_cen': {}}

    # area_info_swt['area_cen']['edges'] = [] # include switches that are opened i.e., [['54', '94'], ['151', '300']]
    area_info_swt['area_cen']['edges'] = open_switches
    area_info_swt['area_cen']['source_bus'] = source_bus
    area_info_swt['area_cen']['vsrc'] = v_source
    area_source_bus = area_info_swt['area_cen']['source_bus']

    areas_info = {'bus_info': {}, 'branch_info': {}}
    areas_info['bus_info']['area_cen'] = bus_info
    areas_info['branch_info']['area_cen'] = branch_info

    # List of sub-graphs. The one that has no source_bus is the disconnected one
    sp_graph = list(nx.connected_components(G))
    for k in sp_graph:
        if source_bus == area_source_bus:
            if source_bus in k:
                area = k
                break
        else:
            if source_bus not in k:
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
                bus_info_area_i[key]['kv'] = bus_info[key]['kv']
                bus_info_area_i[key]['s_rated'] = (
                    bus_info[key]['pv'][0][0] + bus_info[key]['pv'][1][0] + bus_info[key]['pv'][2][0])
                bus_info_area_i[key]['pv'] = [
                    [pv[0] * mult_pv, pv[1] * mult_pv] for pv in bus_info[key]['pv']]
                bus_info_area_i[key]['pq'] = [
                    [pq[0] * mult_load, pq[1] * mult_load] for pq in bus_info[key]['pq']]
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
                bus_info_area_i[key]['kv'] = bus_info[key]['kv']
                bus_info_area_i[key]['pv'] = [
                    i * mult_sec_pv for i in bus_info[key]['pv']]
                bus_info_area_i[key]['pq'] = [
                    i * mult_load for i in bus_info[key]['pq']]
                bus_info_area_i[key]['s_rated'] = (bus_info[key]['pv'][0])
                sump += bus_info_area_i[key]['pq'][0]
                sumq += bus_info_area_i[key]['pq'][1]
                sumpv += bus_info_area_i[key]['pv'][0]
                idx += 1
    idx = 0

    secondary_model = ['SPLIT_PHASE', 'TPX_LINE']
    branch_sw_data_area_i = {}
    nor_open = ['sw7', 'sw8']
    for key, val_bus in branch_info.items():
        if val_bus['fr_bus'] in bus_info_area_i and val_bus['to_bus'] in bus_info_area_i:
            if branch_info[key]['type'] not in secondary_model and key not in nor_open:
                branch_sw_data_area_i[key] = {}
                branch_sw_data_area_i[key]['idx'] = idx
                branch_sw_data_area_i[key]['type'] = branch_info[key]['type']
                branch_sw_data_area_i[key]['from'] = bus_info_area_i[branch_info[key]
                                                                     ['fr_bus']]['idx']
                branch_sw_data_area_i[key]['to'] = bus_info_area_i[branch_info[key]
                                                                   ['to_bus']]['idx']
                branch_sw_data_area_i[key]['fr_bus'] = branch_info[key]['fr_bus']
                branch_sw_data_area_i[key]['to_bus'] = branch_info[key]['to_bus']
                branch_sw_data_area_i[key]['phases'] = branch_info[key]['phases']
                if branch_info[key]['type'] == 'SPLIT_PHASE':
                    branch_sw_data_area_i[key]['impedance'] = branch_info[key]['impedance']
                    branch_sw_data_area_i[key]['impedance1'] = branch_info[key]['impedance1']
                else:
                    branch_sw_data_area_i[key]['zprim'] = branch_info[key]['zprim']
                idx += 1
    idx = 0
    for key, val_bus in branch_info.items():
        if val_bus['fr_bus'] in bus_info_area_i and val_bus['to_bus'] in bus_info_area_i:
            if branch_info[key]['type'] in secondary_model:
                branch_sw_data_area_i[key] = {}
                branch_sw_data_area_i[key]['idx'] = idx
                branch_sw_data_area_i[key]['type'] = branch_info[key]['type']
                branch_sw_data_area_i[key]['from'] = bus_info_area_i[branch_info[key]
                                                                     ['fr_bus']]['idx']
                branch_sw_data_area_i[key]['to'] = bus_info_area_i[branch_info[key]
                                                                   ['to_bus']]['idx']
                branch_sw_data_area_i[key]['fr_bus'] = branch_info[key]['fr_bus']
                branch_sw_data_area_i[key]['to_bus'] = branch_info[key]['to_bus']
                branch_sw_data_area_i[key]['phases'] = branch_info[key]['phases']
                if branch_info[key]['type'] == 'SPLIT_PHASE':
                    branch_sw_data_area_i[key]['impedance'] = branch_info[key]['impedance']
                    branch_sw_data_area_i[key]['impedance1'] = branch_info[key]['impedance1']
                else:
                    branch_sw_data_area_i[key]['impedance'] = branch_info[key]['zprim']
                idx += 1

    return branch_sw_data_area_i, bus_info_area_i


def check_network_radiality(branch_info_cen, bus_info_cen, bus_info):
    if not len(bus_info_cen)-len(branch_info_cen) == 1:
        logger.debug("Network is not Radial. Please check the network data")
