
import os
import numpy as np
import pandas as pd
import re
import networkx as nx
import logging

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)
def get_regulator_incidence_matrix(node_order, dssobj):
    """ Function used for extracting topology from OpenDSS model and use it for finding regulator position with respect to node from the source bus
    The incidence node bus to branch from opendss information for the model has been given in https://sourceforge.net/p/electricdss/discussion/beginners/thread/13164582/
    This function must be run after the power flow has been solved
    The OPENDSS incidence matrix does not have node (phase) information e.g., it will say bus652, not bus652.1, bus652.2, bus652.3 to show phase a, b, c
    The code extracts node (phase) informations from the listed network nodes and branches
    Eventually, we get a graph, from which we trace whether a transformer exists in the path of a node to the source

    """


    logger.debug(f"developing adjacency matrix")

    dssobj.run_command('calcincmatrix_o')
    dssobj.run_command('Export IncMatrix')        # Exports the incidence branch to node matrix (only the non-zero elements and their coordinates)
    dssobj.run_command('Export IncMatrixRows')    # Exports the name of the rows (link branches)
    dssobj.run_command('Export IncMatrixCols')    # Exports the name of the rows (link branches)
    name = dssobj.Circuit.Name()
    incidence_file_name = name+'_inc_matrix.csv'
    incidence_file_row_name = name+'_inc_matrix_rows.csv'
    incidence_file_col_name = name+'_inc_matrix_cols.csv'
    if os.path.isfile(incidence_file_name):
        dss_incidence_matrix = pd.read_csv(incidence_file_name)
        dss_incidence_rows = pd.read_csv(incidence_file_row_name)
        dss_incidence_columns = pd.read_csv(incidence_file_col_name)
    else:
        logger.debug(r" didn't find the incidence matrices files")

    sys_inc_data = np.zeros(shape=(dss_incidence_rows.shape[0], dss_incidence_columns.shape[0]), dtype=int)

    for i in range(0, dss_incidence_matrix['Row'].size):
        sys_inc_data[dss_incidence_matrix['Row'][i], dss_incidence_matrix['Col'][i]] = int(dss_incidence_matrix['Value'][i])


    # Creating dataframe of the incidence matrix with nodes on the indices and branches (Delivery Elements) on the columns
    sys_incidence_matrix = pd.DataFrame(data=sys_inc_data.T, index=dss_incidence_columns.values,
             columns=dss_incidence_rows.values)

    # The above adjacency matrix does not have node (phase) information e.g., it will say bus652, not bus652.1, bus652.2, bus652.3 to show phase a, b, c
    # extracting node (phase) informations from the listed network nodes and branches
    # branches with phase extraction
    branches_with_nodes = []
    for i in range(0, len(dss_incidence_rows.values)):
        dssobj.Circuit.SetActiveElement(dss_incidence_rows.values[i][0])
        phase_order = dssobj.CktElement.NodeOrder()
        phase_order = [x for x in phase_order if x != 0]
        for j in range(0, int(len(phase_order) / 2)):
            current_branch_with_phase = [dss_incidence_rows.values[i][0] + '.' + str(phase_order[j])]
            branches_with_nodes.append(current_branch_with_phase[0])

    # bus with nodes extraction
    bus_with_nodes = []
    node_idx = []
    node_order_with_lower_case = []
    for k in range(0, len(node_order)):
        node_order_with_lower_case.append(node_order[k].lower())
    for i in range(0, len(dss_incidence_columns.values)-1):
    #for i in range(0, len(dss_incidence_rows.values)-1): # old loop worked for IEEE 13 bus
        dssobj.Circuit.SetActiveBus(dss_incidence_columns.values[i][0])
        nodes = dssobj.Bus.Nodes()  # for buses I could only find node order
        for j in range(0, len(nodes)):
            current_bus_with_nodes = [dss_incidence_columns.values[i][0] + '.' + str(nodes[j])]
            node_idx.append([node_order_with_lower_case.index(k) for k in current_bus_with_nodes][0])
            bus_with_nodes.append(current_bus_with_nodes[0])

    # initializing incidence matrix with nodes information
    incidence_matrix_with_nodes = np.zeros(shape=(len(bus_with_nodes)+1, len(branches_with_nodes)), dtype=int)

    # loop through the system incidence matrix rows
    # find the corresponding columns which are connected (either 1 or -1)
    # find the corresponding ids of the buses and branches with node information
    for i in range(0, len(sys_incidence_matrix.index)):
        idx_connected_branches= []
        bus_connected = sys_incidence_matrix.index[i]
        idx_connected_branches.append(
            [i for i, x in enumerate(sys_incidence_matrix.iloc[i, :].values.tolist()) if (x == 1 or x == -1)])
        branch_connected = []
        branch_idx = []
        for j in range(0, len(idx_connected_branches)):
            branch_connected.append(sys_incidence_matrix.columns[idx_connected_branches[j]].values)
            matching_branches = []

            for i in range(0, len(branch_connected[0])):
                matching_branches.append(list(filter(lambda x: x.startswith(branch_connected[0][i]), branches_with_nodes)))
                for j in range(0, len(matching_branches[i])):
                    branch_idx.append(branches_with_nodes.index(matching_branches[i][j]))

        # idx_with_bus_connected_with_nodes = [i for i, x in enumerate(bus_with_nodes) if x == bus_connected]
        matching_buses = list(filter(lambda x: x.startswith(bus_connected[0]), node_order_with_lower_case))
        bus_node_idx = []
        for i in range(0, len(matching_buses)):
            bus_node_idx.append(node_order_with_lower_case.index(matching_buses[i]))


        # now bus with nodes and its corresponding branches with node information is available
        # assign buses and branches to their appropriate node, i.e.,
        # connect node 1 (phase a) of a bus with node 1 (phase a) of a branch and so on and so forth
        phase_a_buses = []
        phase_b_buses = []
        phase_c_buses = []
        phase_a_branches = []
        phase_b_branches = []
        phase_c_branches = []
        for i in range(0, len(bus_node_idx)):

            if matching_buses[i].endswith('.1'): # phase a of the bus (node 1)
                phase_a_buses.append(matching_buses[i])
                node_id = node_order_with_lower_case.index(matching_buses[i])
                for j in range(0, len(matching_branches)):
                    if len(matching_branches[j]) > 1:
                        for k in range(0, len(matching_branches[j])):
                            if matching_branches[j][k].endswith('.1'):
                                phase_a_branches.append(matching_branches[j][k])
                                branch_id = branches_with_nodes.index(matching_branches[j][k])
                                incidence_matrix_with_nodes[node_id, branch_id] = 1
                    else:
                        if matching_branches[j][0].endswith('.1'):
                            phase_a_branches.append(matching_branches[j])
                            branch_id = branches_with_nodes.index(matching_branches[j][0])
                            incidence_matrix_with_nodes[node_id, branch_id] = 1

            elif matching_buses[i].endswith('.2'):  # phase b of bus (node 2)
                phase_b_buses.append(matching_buses[i])
                node_id = node_order_with_lower_case.index(matching_buses[i])
                for j in range(0, len(matching_branches)):
                    if len(matching_branches[j]) > 1:
                        for k in range(0, len(matching_branches[j])):
                            if matching_branches[j][k].endswith('.2'):
                                phase_b_branches.append(matching_branches[j][k])
                                branch_id = branches_with_nodes.index(matching_branches[j][k])
                                incidence_matrix_with_nodes[node_id, branch_id] = 1
                    else:
                        if matching_branches[j][0].endswith('.2'):
                            phase_b_branches.append(matching_branches[j])
                            branch_id = branches_with_nodes.index(matching_branches[j][0])
                            incidence_matrix_with_nodes[node_id, branch_id] = 1

            elif matching_buses[i].endswith('.3'):  # phase c of the bus (node 3)
                phase_c_buses.append(matching_buses[i])
                node_id = node_order_with_lower_case.index(matching_buses[i])
                for j in range(0, len(matching_branches)):
                    if len(matching_branches[j]) > 1:
                        for k in range(0, len(matching_branches[j])):
                            if matching_branches[j][k].endswith('.3'):
                                phase_c_branches.append(matching_branches[j][k])
                                branch_id = branches_with_nodes.index(matching_branches[j][k])
                                incidence_matrix_with_nodes[node_id, branch_id] = 1
                    else:
                        if matching_branches[j][0].endswith('.3'):
                            phase_c_branches.append(matching_branches[j])
                            branch_id = branches_with_nodes.index(matching_branches[j][0])
                            incidence_matrix_with_nodes[node_id, branch_id] = 1

    # we have the incidence matrix with all nodes and branches now let's make first a dataframe and then convert it
    # to a graph using networkx
    # but adjacency matrix for networkx operation must be a square matrix, otherwise it won't be made a graph,
    # so we will make it square with the extra rows and columns won't have any incidence value (connection) so the graph is still the same in principle
    # sys_incidence_matrix_for_graph = pd.DataFrame(data=incidence_matrix_with_nodes, index=node_order_with_lower_case,
    #          columns=branches_with_nodes)
    #
    # # edges of the graph
    # edges = sys_incidence_matrix_for_graph.columns
    #
    #
    # for i in sys_incidence_matrix_for_graph.index:
    #     sys_incidence_matrix_for_graph[i] = [0 for _ in range(len(sys_incidence_matrix_for_graph.index))]
    #
    # for e in edges:
    #     sys_incidence_matrix_for_graph = sys_incidence_matrix_for_graph.append(pd.Series({c: 0 for c in sys_incidence_matrix_for_graph.columns}, name=e))
        # sys_incidence_matrix_for_graph = pd.concat(pd.Series({c: 0 for c in sys_incidence_matrix_for_graph.columns}, name=e))
    combined_columns = branches_with_nodes + node_order_with_lower_case
    combined_index = node_order_with_lower_case + branches_with_nodes
    sys_incidence_matrix_for_graph = pd.DataFrame(index=combined_index, columns=combined_columns).fillna(0)
    sys_incidence_matrix_for_graph.loc[node_order_with_lower_case, branches_with_nodes] = incidence_matrix_with_nodes

    # more efficient implementation of the above without any loops and the "append" and indexing using list was causing warnings and poor performance

    G = nx.from_pandas_adjacency(sys_incidence_matrix_for_graph) # undirected graph

    # verify the graph
    # figure(figsize=(10, 8))
    # nx.draw_shell(G_dir, with_labels=True)

#        reg_list = [sys_incidence_matrix_for_graph.columns.tolist()[3],sys_incidence_matrix_for_graph.columns.tolist()[36],sys_incidence_matrix_for_graph.columns.tolist()[37]]
#        source_buses = ['sourcebus.1', 'sourcebus.2', 'sourcebus.3']


    reg_names = dssobj.RegControls.AllNames()
    reg_list = []
    reg_taps = []
    reg_control_list = []
# if caseName == 'IEEE123':
    go = dssobj.RegControls.First()
    while go:
        dssobj.Circuit.SetActiveElement(dssobj.RegControls.Name())
        numphases = dssobj.CktElement.NumPhases()
        xfrmr_name = dssobj.RegControls.Transformer()
        phase_order = dssobj.CktElement.NodeOrder()
        for j in range(0, numphases):
            reg_list.append('Transformer.' + xfrmr_name + '.' + str(phase_order[j]))
            reg_taps.append(dssobj.RegControls.TapNumber())
            reg_control_list.append(dssobj.RegControls.Name())
        go = dssobj.RegControls.Next()
# else:
    #     for i in range(0, len(reg_names)):
    #         dssobj.Circuit.SetActiveClass('RegControl')
    #         dssobj.Circuit.SetActiveElement(reg_names[i])
    #         numphases = dssobj.CktElement.NumPhases()
    #         phase_order = dssobj.CktElement.NodeOrder()
    #         xfrmr_name = dssobj.RegControls.Transformer()
    #         for j in range(0, numphases):
    #             reg_list.append('Transformer.'+ xfrmr_name + '.' + str(phase_order[j]))

    source_buses = [each_string.lower() for each_string in node_order[0:3]] # first three nodes are usually the source bus

    dssobj.reg_inc_matrix = np.zeros(shape=(len(node_order), len(reg_list)), dtype=int)
    incidence_column = []
    paths = []
    for k in range(0, len(source_buses)):
        source_bus = source_buses[k]
        for i in range(0, len(node_order)):
            # get the path from node to the source
            reg_found_list = []
            try:
                path = nx.shortest_path(G, source=source_bus, target=sys_incidence_matrix_for_graph.index[i])
                # find a regulator within the path
                reg_found_list = list(set(reg_list) & set(path))
                paths.append(path)

            except:
                logger.debug(f" no path between {source_bus} and {sys_incidence_matrix_for_graph.index[i]}")
            # if regulator found
            if len(reg_found_list) > 0:
                logger.debug(f" regulator {reg_found_list} in path {path}")
                # loop through regulators
                for j in range(0, len(reg_found_list)):
                    incidence_column.append(reg_list.index(reg_found_list[j]))
                    dssobj.reg_inc_matrix[i, reg_list.index(reg_found_list[j])] = 1

    #reg_order_name = reg_list
    reg_control_order_name  = reg_control_list
    reg_inc_matrix = dssobj.reg_inc_matrix[3:, ]  # taking first three rows out as they correspond to the source bus

#     # get nominal taps
#     reg_taps = []
#     reg_names = dssobj.RegControls.AllNames()
#     for i in range(0, len(reg_names)):
#         dssobj.Circuit.SetActiveClass('RegControl')
#         dssobj.Circuit.SetActiveElement(reg_names[i])
#         # phase_order = dssobj.CktElement.NodeOrder()
#         # xfrmr_name = dssobj.RegControls.Transformer()
#         numphases = dssobj.CktElement.NumPhases()
#         for j in range(0, numphases):
#             reg_taps.append(dssobj.RegControls.TapNumber())
# #            reg_list.append('Transformer.' + xfrmr_name + '.' + str(phase_order[j]))
    reg_taps = np.array(reg_taps)
    return reg_control_order_name, reg_taps, reg_inc_matrix


def get_flex_load_incidence_matrix(node_order, dssobj, u_idx, pL):


    flex_load_inc_matrix = np.zeros(shape=(len(node_order), len(u_idx)), dtype=int)

    find = re.compile(r"^[^.]*")
    flex_load_nodes = []
    node_idx = []
    load_names = []
    for j in range(0, len(u_idx)):
        load_name = u_idx[j]
        load_names.append(load_name)
        dssobj.Circuit.SetActiveClass('Load')
        dssobj.Circuit.SetActiveElement(load_name)
        temp = dssobj.CktElement.BusNames()
        connected_bus = re.search(find, temp[0]).group(0)
        phase_order = dssobj.CktElement.NodeOrder()
        for i in range(0, len(phase_order) - 1):
            current_node = [connected_bus + '.' + str(phase_order[i])]
            node_idx.append([node_order.index(i) for i in current_node][0])
            flex_load_nodes.append(current_node)

        for j in range(0, len(node_idx)):
            flex_load_inc_matrix[node_idx[j], j] = 1

    flex_load_inc_matrix = flex_load_inc_matrix[3:, ]
    # flexible load vector in the form of grid nodes

    p_flex_load_init = flex_load_inc_matrix.T.dot(pL)

    return load_names, p_flex_load_init, flex_load_inc_matrix

def get_caps_incidence_matrix(node_order, dssobj, capbk_df):

    # u_cap = ['Cap1', 'Cap2', 'Cap3', 'Cap4']
    cap_names = capbk_df['Name']
    cap_length = len(cap_names)
    cap_load_inc_matrix = np.zeros(shape=(len(node_order), cap_length), dtype=int)

    find = re.compile(r"^[^.]*")
    cap_load_nodes = []
    node_idx = []
    for j in range(0, cap_length):
        cap_name = cap_names[j]
        dssobj.Circuit.SetActiveClass('Capacitor')
        dssobj.Circuit.SetActiveElement(cap_name)
        temp = dssobj.CktElement.BusNames()
        connected_bus = re.search(find, temp[0]).group(0)
        phase_order = dssobj.CktElement.NodeOrder()
        for i in range(0, len(phase_order) - 1):
            current_node = [connected_bus + '.' + str(phase_order[i])]
            node_idx.append([node_order.index(i) for i in current_node][0])
            cap_load_nodes.append(current_node)
        for j in range(0, len(node_idx)):
            cap_load_inc_matrix[node_idx[j], j] = 1

    cap_load_inc_matrix = cap_load_inc_matrix[3:, ]

    cap_bank_init = capbk_df['kvar'].values * 1000 # 1000 for kvar to var

    return cap_names, cap_bank_init, cap_load_inc_matrix


def get_pv_incidence_matrix(node_order, dssobj, pvs_df):

    pv_names = pvs_df['Name'].values
    pv_length = len(pv_names)
    pv_inc_matrix = np.zeros(shape=(len(node_order), pv_length), dtype=int)

    find = re.compile(r"^[^.]*")
    pv_nodes = []
    node_idx = []
    for j in range(0, pv_length):
        pv_name = pv_names[j]
        dssobj.Circuit.SetActiveClass('PVSystem')
        dssobj.Circuit.SetActiveElement(pv_name)
        temp = dssobj.CktElement.BusNames()
        connected_bus = re.search(find, temp[0]).group(0)
        phase_order = dssobj.CktElement.NodeOrder()
        for i in range(0, len(phase_order) - 1):
            current_node = [connected_bus + '.' + str(phase_order[i])]
            node_idx.append([node_order.index(i) for i in current_node][0])
            pv_nodes.append(current_node)
        for j in range(0, len(node_idx)):
            pv_inc_matrix[node_idx[j], j] = 1

    pv_inc_matrix = pv_inc_matrix[3:, ]

    pv_q_init = pvs_df['kvar'].values * 1000 # 1000 for kvar to var
    pv_p_init = pvs_df['kW'].values * 1000 # 1000 for kW to var
    pv_s_init = pvs_df['kVARated'].values * 1000 # 1000 for kVA
    return pv_names, pv_p_init, pv_q_init, pv_s_init, pv_inc_matrix