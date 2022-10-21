#from scipy.sparse import csc_matrix
from scipy import sparse as sparse
import numpy as np
import re

import logging
logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.DEBUG)

def snapshot_run(dssobj):
    """ Function used for getting the snapshot run result
    """
    logger.debug(f"Solving OpenDSS PowerFlow in Snapshot Mode")
    dssobj.run_command('set mode=snap')
    dssobj.run_command('solve')
    logger.debug(f"solution converged? {dssobj.Solution.Converged()}")
    logger.debug(f"number of iterations took: {dssobj.Solution.Iterations()}")
    # self.dss_obj.run_command('Show convergence')
    # self.dss_obj.Solution.Solve()

def get_y_matrix(dssobj, self):
    """ Function used for extracting system Y matrix from OpenDSS model
    This gets the Ybus without load impedances and source impedances. This is important for linearization procedure
    """

    logger.debug(f"Turning off all impedance except lines for getting constant P-Q load dependent Y bus")

    dssobj.run_command('batchedit regcontrol..* enabled=false')
    dssobj.run_command('batchedit vsource..* enabled=false')
    dssobj.run_command('batchedit isource..* enabled=false')
    dssobj.run_command('batchedit load..* enabled=false')
    dssobj.run_command('batchedit generator..* enabled=false')
    dssobj.run_command('batchedit pvsystem..* enabled=false')
    dssobj.run_command('batchedit storage..* enabled=false')
    dssobj.run_command('CalcVoltageBases')
    dssobj.run_command('set maxiterations=20')

    dssobj.run_command('solve')
    dssobj.run_command('export Y')

    # i don't know but this was not working for what reason in the debug mode. Only in run mode. So I wrote a little code for it
    # y_matrix = csc_matrix(dssobj.YMatrix.getYsparse())

    # return Ymatrix

    ysparsedata = dssobj.YMatrix.getYsparse()

    data = ysparsedata[0]
    indices= ysparsedata[1]
    indptr =ysparsedata[2]
    y_matrix = np.ndarray((self._opf_num_nodes, self._opf_num_nodes), dtype=np.complex_)

    for i in range(0, self._opf_num_nodes):
        #print(i)
        k = indptr[i]
        l = indptr[i + 1]
        y_matrix[i, indices[k:l]] = data[k:l]

    Y_LL = y_matrix[3:, 3:]
    Y_L0 = y_matrix[3:, 0:3]
    Y_0L = y_matrix[0:3, 3:]
    Y_00 = y_matrix[0:3, 0:3]
    logger.debug(f"Putting them back on and solving them to compare results")

    dssobj.run_command('batchedit regcontrol..* enabled=true')
    dssobj.run_command('batchedit vsource..* enabled=true')
    dssobj.run_command('batchedit isource..* enabled=true')
    dssobj.run_command('batchedit load..* enabled=true')
    dssobj.run_command('batchedit generator..* enabled=true')
    dssobj.run_command('batchedit pvsystem..* enabled=true')
    dssobj.run_command('batchedit storage..* enabled=true')
    snapshot_run(dssobj)
    return y_matrix, Y_LL, Y_L0, Y_0L, Y_0L, Y_00
#        return Ysparse

def get_voltage_vector(dssobj):
    """
    Function used for extracting Voltage phasor information at all nodes from OpenDSS model
    The order of voltage is similar to what Y-bus uses
    """
    vol = dssobj.Circuit.YNodeVArray()
    v_real = np.array([vol[i] for i in range(len(vol)) if i % 2 == 0])
    v_imag = np.array([vol[i] for i in range(len(vol)) if i % 2 == 1])
    voltages = v_real + np.multiply(1j, v_imag)
    v0 = voltages[0:3]
    vL = voltages[3:]
    # self.v0 = self.voltages[0:3]
    # self.vL = self.voltages[3:]
    # print(self.vL)
    # return V
    return voltages, v0, vL


def get_load_vector(self, dss_obj):
    """
    Function used for extracting load array corresponding to node order
    which is similar to what Y-bus node order is
    """
    find = re.compile(r"^[^.]*")
    load_nodes = []
    node_idx = []
    load_kW = []
    load_kVAR = []
    # print(f'{np.__version__}')
    # print(f'{sp.__version__}')
    for j in range(0, len(self._opf_loads_df)):
        load_name = self._opf_loads_df.iloc[j]['Name']

        dss_obj.Circuit.SetActiveClass('Load')
        dss_obj.Circuit.SetActiveElement(load_name)
        temp = dss_obj.CktElement.BusNames()
        connected_bus = re.search(find, temp[0]).group(0)
        phase_order = dss_obj.CktElement.NodeOrder()
        load_kw_val = dss_obj.CktElement.TotalPowers()[0]
        load_kvar_val = dss_obj.CktElement.TotalPowers()[1]
        for i in range(0, len(phase_order) - 1):
            current_node = [connected_bus + '.' + str(phase_order[i])]
            # TODO: This makes it O(n^2)
            node_idx.append([self._opf_node_order.index(i) for i in current_node][0])
            load_nodes.append(current_node)
            load_kW.append(load_kw_val/(len(phase_order)-1))
            load_kVAR.append(load_kvar_val/(len(phase_order)-1))
            # load_kW.append(self._opf_loads_df.iloc[j]['kW']/(len(phase_order)-1))
            # load_kVAR.append(self._opf_loads_df.iloc[j]['kvar']/(len(phase_order)-1))


    p_Y = np.zeros(len(self._opf_node_order))
    q_Y = np.zeros(len(self._opf_node_order))

    p_Y[node_idx] = np.array(load_kW)*1000 # converting into Watts
    q_Y[node_idx] = np.array(load_kVAR)*1000 # converting into Vars

    pL = p_Y[3:]
    qL = q_Y[3:]
    logger.debug(f"====system load vector acquired====")
    return p_Y, q_Y, pL, qL

def get_pv_vector(self, dss_obj):
    """
    Function used for extracting load array corresponding to node order
    which is similar to what Y-bus node order is
    """
    find = re.compile(r"^[^.]*")
    pv_nodes = []
    node_idx = []
    pV_kW = []
    pV_kVAR = []
    # print(f'{np.__version__}')
    # print(f'{sp.__version__}')
    for j in range(0, len(self._opf_pvs_df)):
        pv_name = self._opf_pvs_df.iloc[j]['Name']

        dss_obj.Circuit.SetActiveClass('PVSystem')
        dss_obj.Circuit.SetActiveElement(pv_name)
        temp = dss_obj.CktElement.BusNames()
        connected_bus = re.search(find, temp[0]).group(0)
        phase_order = dss_obj.CktElement.NodeOrder()
        pv_kW_val = -dss_obj.CktElement.TotalPowers()[0]
        pv_kvar_val = -dss_obj.CktElement.TotalPowers()[1]

        for i in range(0, len(phase_order) - 1):
            current_node = [connected_bus + '.' + str(phase_order[i])]
            node_idx.append([self._opf_node_order.index(i) for i in current_node][0])
            pv_nodes.append(current_node)
            pV_kW.append(pv_kW_val / (len(phase_order) - 1))
            pV_kVAR.append(pv_kvar_val / (len(phase_order) - 1))

            # pV_kW.append(self._opf_pvs_df.iloc[j]['kW'] / (len(phase_order) - 1))
            # pV_kVAR.append(self._opf_pvs_df.iloc[j]['kvar'] / (len(phase_order) - 1))

    p_pV_Y = np.zeros(len(self._opf_node_order))
    q_pV_Y = np.zeros(len(self._opf_node_order))
    # converting into Watts, Vars
    p_pV_Y[node_idx] = np.array(pV_kW) * 1000
    q_pV_Y[node_idx] = np.array(pV_kVAR) * 1000

    p_pV_L = p_pV_Y[3:]
    q_pV_L = q_pV_Y[3:]
    logger.debug(f"====system pv vector acquired ====")
    return p_pV_Y, q_pV_Y, p_pV_L, q_pV_L


def get_q_cap_vector(self, dss_obj):
    """
    Function used for extracting load array corresponding to node order
    which is similar to what Y-bus node order is
    """
    find = re.compile(r"^[^.]*")
    cap_nodes = []
    node_idx = []
    q_cap_kVAR = []
    # print(f'{np.__version__}')
    # print(f'{sp.__version__}')
    for j in range(0, len(self._opf_capbk_df)):
        cap_name = self._opf_capbk_df.iloc[j]['Name']

        dss_obj.Circuit.SetActiveClass('Capacitor')
        dss_obj.Circuit.SetActiveElement(cap_name)
        temp = dss_obj.CktElement.BusNames()
        connected_bus = re.search(find, temp[0]).group(0)
        phase_order = dss_obj.CktElement.NodeOrder()
        kvar_val = abs(dss_obj.CktElement.TotalPowers()[1])
        for i in range(0, int(len(phase_order)/2)):
            current_node = [connected_bus + '.' + str(phase_order[i])]
            node_idx.append([self._opf_node_order.index(i) for i in current_node][0])
            cap_nodes.append(current_node)
            q_cap_kVAR.append(kvar_val / (len(phase_order)/2))

            # q_cap_kVAR.append(self._opf_capbk_df.iloc[j]['kvar'] / (len(phase_order) - 1))

    q_cap_Y = np.zeros(len(self._opf_node_order))

    q_cap_Y[node_idx] = np.array(q_cap_kVAR) * 1000

    q_cap_L = q_cap_Y[3:]
    logger.debug(f"====system cap vector acquired ====")
    return q_cap_Y, q_cap_L


def get_taps_vector(self, dssobj):
    reg_list = []
    reg_taps = []
    reg_control_list = []
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

    reg_taps = np.array(reg_taps)
    return reg_taps

def xfrmr_data_fcn(dssobj):
    Bus1 = []
    Bus2 = []
    NumPhases = []
    y_matrix = []
    EmergAmp = []
    for Xfrmr in dssobj.Transformers.AllNames():
        dssobj.Circuit.SetActiveElement('Transformer.' + Xfrmr)
        Bus1.append(dssobj.CktElement.BusNames()[0])
        Bus2.append(dssobj.CktElement.BusNames()[1])
        NumPhases.append(dssobj.CktElement.NumPhases())
        el_node_order = dssobj.CktElement.NodeOrder()
        phases = len(el_node_order) / 2
        # print(f'numpy version-->{np.__version__}')
        # print(f'scipy version-->{sp.__version__}')
        # print(f'dss version-->{dssobj.__version__}')
        # y_temp = csc_matrix(
        #     np.reshape(np.array(dssobj.CktElement.YPrim()), (int(phases) * 4, int(phases) * 2), order="F").transpose())
        y_temp = np.reshape(np.array(dssobj.CktElement.YPrim()), (int(phases) * 4, int(phases) * 2), order="F").transpose()
        real_ids = np.arange(0, y_temp.shape[1], 2)
        imag_ids = np.arange(1, y_temp.shape[1], 2)
        y_matrix.append(y_temp[:, real_ids] + np.multiply(1j, y_temp[:, imag_ids]))
        EmergAmp.append(dssobj.CktElement.EmergAmps())

    xfrmr_data = {'Bus1': Bus1, 'Bus2': Bus2, 'Phases': NumPhases, 'Yprim': y_matrix, 'EmergAmps': EmergAmp}
    return xfrmr_data


def lines_data_fcn(dssobj):
    Bus1 = []
    Bus2 = []
    NumPhases = []
    y_matrix = []
    EmergAmp = []
    for line in dssobj.Lines.AllNames():
        dssobj.Circuit.SetActiveElement('Line.' + line)
        Bus1.append(dssobj.CktElement.BusNames()[0])
        Bus2.append(dssobj.CktElement.BusNames()[1])
        NumPhases.append(dssobj.CktElement.NumPhases())
        el_node_order = dssobj.CktElement.NodeOrder()
        phases = len(el_node_order) / 2
        # y_temp = csc_matrix(
        #     np.reshape(np.array(dssobj.CktElement.YPrim()), (int(phases) * 4, int(phases) * 2), order="F").transpose())
        y_temp = np.reshape(np.array(dssobj.CktElement.YPrim()), (int(phases) * 4, int(phases) * 2), order="F").transpose()
        real_ids = np.arange(0, y_temp.shape[1], 2)
        imag_ids = np.arange(1, y_temp.shape[1], 2)
        y_matrix.append(y_temp[:, real_ids] + np.multiply(1j, y_temp[:, imag_ids]))
        EmergAmp.append(dssobj.CktElement.EmergAmps())

    lines_data = {'Bus1': Bus1, 'Bus2': Bus2, 'Phases': NumPhases, 'Yprim': y_matrix, 'EmergAmps': EmergAmp}
    return lines_data


def line_flow_from_nodes_matrix(dssobj):
    """
    Function to extract line flow "from nodes" matrix so that line flow currents can be calculated as
    i_f = Y_f * v
    where
    i_f are the line flow current from
    Y_f is the line flow from admittance matrix
    v are the nodal voltages.
    """
    #TODO: fix string finding when 2 phases are connected
    #if dssobj.caseName == 'IEEE123':
    nlines = len(dssobj._opf_lines_data['Bus1'])
    nlines_with_phases = np.array(dssobj._opf_lines_data['Phases']).sum()

    # else:
    #     nlines = dssobj.lines_df.shape[0]
    #     nlines_with_phases = dssobj.lines_df['Phases'].sum()

    nxfrmr = len(dssobj._opf_xfrmr_data)#.shape[0]
    # nlines_with_phases = dssobj.lines_df['Phases'].sum()
    nxfrmr_with_phases = np.array(dssobj._opf_xfrmr_data['Phases']).sum()

    # line_flow_from_matrix = csc_matrix((nlines_with_phases + nxfrmr_with_phases, dssobj._opf_num_nodes),
    #                                         dtype=np.complex128).toarray()
    line_flow_from_matrix = np.ndarray((nlines_with_phases + nxfrmr_with_phases, dssobj._opf_num_nodes),
                                            dtype=np.complex128)
    dssobj.line_ratings = np.zeros(shape=(nlines_with_phases+nxfrmr_with_phases))

    # find node ids specific to lines
    # set_of_nodes = set(self.nodes.tolist())
    # set_of_nodes = [x.lower() for x in self._opf_node_order.tolist()]
    set_of_nodes = [x.lower() for x in dssobj._opf_node_order]

    # this is for extracting line admittances from line data information
    line_index = 0
    # if dssobj.caseName == 'IEEE123': # something crazy is happening in IEEE123 dataframe format, because the line codes are in single phase, but the dataframe shows as 2 phases
    #     dssobj.lines_df['Phases'][28] = 1
    #     dssobj.lines_df['Phases'][29] = 1
    #     dssobj.lines_df['Phases'][37] = 1
    #     dssobj.lines_df['Phases'][38] = 1
    find = re.compile(r"^[^.]*")

# if dssobj.caseName == 'IEEE123':
    for n in range(0, nlines):
        from_nodes = []
        to_nodes = []
        numphases = dssobj._opf_lines_data['Phases'][n]
        if numphases == 3:
            temp = dssobj._opf_lines_data['Bus1'][n]
            connected_bus_from = re.search(find, temp).group(0)
            temp = dssobj._opf_lines_data['Bus2'][n]
            connected_bus_to = re.search(find, temp).group(0)
            for k in range(0, numphases):
                from_nodes.append(connected_bus_from + '.' + str(k + 1))
                to_nodes.append(connected_bus_to + '.' + str(k + 1))

            # find node ids specific to lines
            # set_of_nodes = set(self.nodes.tolist())

            from_nodes = [x.lower() for x in from_nodes]
            to_nodes = [x.lower() for x in to_nodes]
            from_indices = [i for i, item in enumerate(set_of_nodes) if item in from_nodes]
            to_indices = [i for i, item in enumerate(set_of_nodes) if item in to_nodes]
        else:  # there could be any sequence of nodes so this part of is not the generic code
            temp_from_node = dssobj._opf_lines_data['Bus1'][n]
            temp_to_node = dssobj._opf_lines_data['Bus2'][n]
            temp_from_node_id = temp_from_node.index('.')
            temp_to_node_id = temp_to_node.index('.')
            try:
                from_node_id = temp_from_node[0:temp_from_node_id]
                to_node_id = temp_to_node[0:temp_to_node_id]

                for k in range(0, numphases):
                    from_nodes.append(from_node_id + '.' + temp_from_node[temp_from_node_id + 1])
                    to_nodes.append(to_node_id + '.' + temp_to_node[temp_to_node_id + 1])
                    temp_from_node_id += 2
                    temp_to_node_id += 2
                # from_indices = [i for i, item in enumerate(set_of_nodes) if item in from_nodes]
                # to_indices = [i for i, item in enumerate(set_of_nodes) if item in to_nodes]
            except:  # if index not found from ".index()" method above then it will throw an error, we should then just avoid it
                logger.debug(
                    f"Couldn't locate line connected between Bus {dssobj._opf_lines_data['Bus1']} and {dssobj._opf_lines_data['Bus2']}")
                pass

            from_indices = [i for i, item in enumerate(set_of_nodes) if item in from_nodes]
            to_indices = [i for i, item in enumerate(set_of_nodes) if item in to_nodes]

        if len(from_indices) > 0 and len(to_indices) > 0:
            line_flow_from_matrix[line_index:line_index + len(from_indices), from_indices] \
                = line_flow_from_matrix[line_index:line_index + len(from_indices), from_indices] \
                  + dssobj._opf_lines_data['Yprim'][n][0:len(from_indices), 0:len(from_indices)]
            line_flow_from_matrix[line_index:line_index + len(from_indices), to_indices] \
                = line_flow_from_matrix[line_index:line_index + len(from_indices), to_indices] \
                  + dssobj._opf_lines_data['Yprim'][n][0:len(from_indices),
                    len(from_indices):len(from_indices) + 1 + len(to_indices)]
            dssobj.line_ratings[line_index:line_index + len(from_indices)] = dssobj._opf_lines_data['EmergAmps'][
                                                                                 n] / len(from_indices)
        else:
            logger.debug(f"couldn't find nodes for line connected between bus {dssobj._opf_lines_data['Bus1'][n]} and bus {dssobj._opf_lines_data['Bus2'][n]}")
        line_index += len(from_indices)
    # else:
    #     for n in range(0, nlines):
    #         from_nodes = []
    #         to_nodes = []
    #
    #         numphases = dssobj.lines_df['Phases'][n]
    #         if numphases == 3:
    #             for k in range(0, numphases):
    #                 from_nodes.append(dssobj.lines_df['Bus1'][n] + '.' + str(k + 1))
    #                 to_nodes.append(dssobj.lines_df['Bus2'][n] + '.' + str(k + 1))
    #             # just lowering the node names just in case
    #             from_nodes = [x.lower() for x in from_nodes]
    #             to_nodes = [x.lower() for x in to_nodes]
    #
    #             #               set_of_nodes = set(self.nodes.tolist())
    #             # from_indices = [i for i, item in enumerate(set_of_nodes) if item in from_nodes]
    #             # to_indices = [i for i, item in enumerate(set_of_nodes) if item in to_nodes]
    #         else: # there could be any sequence of nodes so this part of is not the generic code
    #             temp_from_node = dssobj.lines_df['Bus1'][n]
    #             temp_to_node = dssobj.lines_df['Bus2'][n]
    #             temp_from_node_id = temp_from_node.index('.')
    #             temp_to_node_id = temp_to_node.index('.')
    #             try:
    #                 from_node_id = temp_from_node[0:temp_from_node_id]
    #                 to_node_id = temp_to_node[0:temp_to_node_id]
    #
    #                 for k in range(0, numphases):
    #                     from_nodes.append(from_node_id + '.' + temp_from_node[temp_from_node_id+1])
    #                     to_nodes.append(to_node_id + '.' + temp_to_node[temp_to_node_id+1])
    #                     temp_from_node_id += 2
    #                     temp_to_node_id += 2
    #                 # from_indices = [i for i, item in enumerate(set_of_nodes) if item in from_nodes]
    #                 # to_indices = [i for i, item in enumerate(set_of_nodes) if item in to_nodes]
    #             except: # if index not found from ".index()" method above then it will throw an error, we should then just avoid it
    #                 logger.debug(f"Couldn't locate line number {n} connected between Bus {dssobj.lines_df['Bus1'][n]} and {dssobj.lines_df['Bus2'][n]}")
    #                 pass
    #
    #         from_indices = [i for i, item in enumerate(set_of_nodes) if item in from_nodes]
    #         to_indices = [i for i, item in enumerate(set_of_nodes) if item in to_nodes]
    #
    #         if len(from_indices) > 0 and len(to_indices) > 0:
    #             # constructing line admittance matrix
    #             #if dssobj.caseName == 'IEEE123':
    #             #    if n == 28 or n == 29 or n == 37 or n == 38: # something is happening with these line numbers
    #             #         arr = np.array(dssobj.lines_df['Yprim'][n])
    #             #         real_ids = np.arange(0, len(arr), 2)
    #             #         imag_ids = np.arange(1, len(arr), 2)
    #             #
    #             #         y_real = arr[real_ids].reshape(numphases * 4, numphases * 4)
    #             #         y_imag = arr[imag_ids].reshape(numphases * 4, numphases * 4)
    #             #
    #             #         y_line = y_real + np.multiply(1j, y_imag)
    #             #         line_flow_from_matrix[line_index:line_index + len(from_indices), from_indices] \
    #             #             = line_flow_from_matrix[line_index:line_index + len(from_indices), from_indices] \
    #             #               + y_line[0:len(from_indices), 0:len(from_indices)]
    # #                else:
    #             arr = np.array(dssobj.lines_df['Yprim'][n])
    #             real_ids = np.arange(0, len(arr), 2)
    #             imag_ids = np.arange(1, len(arr), 2)
    #
    #             y_real = arr[real_ids].reshape(numphases * 2, numphases * 2)
    #             y_imag = arr[imag_ids].reshape(numphases * 2, numphases * 2)
    #             y_line = y_real + np.multiply(1j, y_imag)
    #             line_flow_from_matrix[line_index:line_index + len(from_indices), from_indices] \
    #                 = line_flow_from_matrix[line_index:line_index + len(from_indices), from_indices] \
    #                   + y_line[0:len(from_indices), 0:len(from_indices)]
    #             line_flow_from_matrix[line_index:line_index + len(from_indices), to_indices] \
    #                 = line_flow_from_matrix[line_index:line_index + len(from_indices), to_indices] \
    #                   + y_line[0:len(from_indices), len(from_indices):]
    #         else:
    #             logger.debug(f"y no line between node {from_nodes} and {to_nodes}")
    #
    #             # arr = np.array(dssobj.lines_df['Yprim'][n])
    #             # real_ids = np.arange(0, len(arr), 2)
    #             # imag_ids = np.arange(1, len(arr), 2)
    #             #
    #             # y_real = arr[real_ids].reshape(numphases * 2, numphases * 2)
    #             # y_imag = arr[imag_ids].reshape(numphases * 2, numphases * 2)
    #             # y_line = y_real + np.multiply(1j, y_imag)
    #             # line_flow_from_matrix[line_index:line_index + len(from_indices), from_indices] \
    #             #     = line_flow_from_matrix[line_index:line_index + len(from_indices), from_indices] \
    #             #       + y_line[0:len(from_indices), 0:len(from_indices)]
    #
    #             # if dssobj.caseName == 'IEEE123':
    #             #     try:
    #             # line_flow_from_matrix[line_index:line_index + len(from_indices), to_indices] \
    #             #     = line_flow_from_matrix[line_index:line_index + len(from_indices), to_indices] \
    #             #       + y_line[0:len(from_indices), len(from_indices):]
    #                 # except:
    #                 #     logger.debug(f"y primitive matrix didn't fit for line connected between node {from_nodes} and {to_nodes}")
    #                 #
    #                 #     line_flow_from_matrix[line_index:line_index + len(from_indices), to_indices] \
    #                 #         = line_flow_from_matrix[line_index:line_index + len(from_indices), to_indices] \
    #                 #           + y_line[0:len(from_indices), len(from_indices):-2]
    #                 #     pass
    #             # else:
    #             #     line_flow_from_matrix[line_index:line_index + len(from_indices), to_indices] \
    #             #         = line_flow_from_matrix[line_index:line_index + len(from_indices), to_indices] \
    #             #           + y_line[0:len(from_indices), len(from_indices):]
    #             dssobj.line_ratings[line_index:line_index+len(from_indices)] = dssobj.lines_df['EmergAmps'][n]/len(from_indices)
    #         line_index += len(from_indices)

    # this is for extracting line admittances from transformer data information
    for n in range(0, nxfrmr):
        from_nodes = []
        to_nodes = []
        numphases = dssobj._opf_xfrmr_data['Phases'][n]
        if numphases == 3:
            for k in range(0, numphases):
                from_nodes.append(dssobj._opf_xfrmr_data['Bus1'][n] + '.' + str(k + 1))
                to_nodes.append(dssobj._opf_xfrmr_data['Bus2'][n] + '.' + str(k + 1))

        # find node ids specific to lines
        # set_of_nodes = set(self.nodes.tolist())

            from_nodes = [x.lower() for x in from_nodes]
            to_nodes = [x.lower() for x in to_nodes]
            from_indices = [i for i, item in enumerate(set_of_nodes) if item in from_nodes]
            to_indices = [i for i, item in enumerate(set_of_nodes) if item in to_nodes]
        else:    # there could be any sequence of nodes so this part of is not the generic code
            temp_from_node = dssobj._opf_xfrmr_data['Bus1'][n]
            temp_to_node = dssobj._opf_xfrmr_data['Bus2'][n]
            temp_from_node_id = temp_from_node.index('.')
            temp_to_node_id = temp_to_node.index('.')
            try:
                from_node_id = temp_from_node[0:temp_from_node_id]
                to_node_id = temp_to_node[0:temp_to_node_id]

                for k in range(0, numphases):
                    from_nodes.append(from_node_id + '.' + temp_from_node[temp_from_node_id+1])
                    to_nodes.append(to_node_id + '.' + temp_to_node[temp_to_node_id+1])
                    temp_from_node_id += 2
                    temp_to_node_id += 2
                # from_indices = [i for i, item in enumerate(set_of_nodes) if item in from_nodes]
                # to_indices = [i for i, item in enumerate(set_of_nodes) if item in to_nodes]
            except: # if index not found from ".index()" method above then it will throw an error, we should then just avoid it
                logger.debug(f"Couldn't locate line connected between Bus {dssobj.lines_df['Bus1']} and {dssobj.lines_df['Bus2']}")
                pass



            from_indices = [i for i, item in enumerate(set_of_nodes) if item in from_nodes]
            to_indices = [i for i, item in enumerate(set_of_nodes) if item in to_nodes]


        if len(from_indices) > 0 and len(to_indices) > 0:
            line_flow_from_matrix[line_index:line_index + len(from_indices), from_indices] \
                = line_flow_from_matrix[line_index:line_index + len(from_indices), from_indices] \
                  + dssobj._opf_xfrmr_data['Yprim'][n][0:len(from_indices), 0:len(from_indices)]
            line_flow_from_matrix[line_index:line_index + len(from_indices), to_indices] \
                = line_flow_from_matrix[line_index:line_index + len(from_indices), to_indices] \
                  + dssobj._opf_xfrmr_data['Yprim'][n][0:len(from_indices),
                    len(from_indices) + 1:len(from_indices) + 1 + len(to_indices)]
            dssobj.line_ratings[line_index:line_index + len(from_indices)] = dssobj._opf_xfrmr_data['EmergAmps'][n]/len(from_indices)
        line_index += len(from_indices)

    zero_rows = any(abs(line_flow_from_matrix).sum(axis=1) == 0)
    # zero_rows = np.where(line_flow_from_matrix.any(axis=1))[0]
    logger.debug(f"line flow admittance has {zero_rows} empty rows: 0 is good, nonzero empty row usually means we missed some lines")

    #Yf_0 = line_flow_from_matrix[:, 0:3]
    #Yf_L = line_flow_from_matrix[:, 3:]
    return line_flow_from_matrix


# verification script for power flows
def get_variables_manual(simobj, dss):

    # simobj.i_f = simobj.Yf_0.dot(simobj.v0) + simobj.Yf_L.dot(simobj.vL)  # line flow from complex
    simobj.i_f = simobj._y_line[:, 0:3].dot(simobj._voltages[0:3]) + simobj._y_line[:, 3:].dot(simobj._voltages[3:])  # line flow from complex

    simobj.i_f_abs = np.abs(simobj.i_f)  # absolute flows complex
    # simobj.s_loss_manual = np.conj(np.conj(simobj.voltages.T).dot(simobj.y_matrix.toarray()).dot(simobj.voltages))
    simobj.s_loss_manual = np.conj(np.conj(simobj._voltages.T).dot(simobj._y_matrix).dot(simobj._voltages))

    simobj.p_loss_dss = dss.Circuit.Losses()[0]
    simobj.q_loss_dss = dss.Circuit.Losses()[1]
    simobj.s_loss_dss = simobj.p_loss_dss + 1j*simobj.q_loss_dss
    simobj.v0_abs_dss = abs(simobj._voltages[0:3])
    simobj.vL_abs_dss = abs(simobj._voltages[3:])
    simobj.i_f_abs_dss = simobj.i_f_abs
    logger.debug(f"Losses from OpenDSS: {simobj.s_loss_dss}")
    logger.debug(f"Losses from Manual Calculations method: {simobj.s_loss_manual} ")




def set_feeder_control_vars(dss, opf_vals_dict):

    powers_P_values = opf_vals_dict['powers_P'].array
    powers_P_list = opf_vals_dict['powers_P'].unique_ids

    for j in range(0, len(powers_P_values)):
        load_name = powers_P_list[j]
        load_kW = powers_P_values[j]/1e3
        change_load_string1 = 'edit load.' + str(load_name) + ' kW' + ' = ' + str(load_kW)
        logger.debug(f"{change_load_string1}")
        dss.run_command(change_load_string1)

    for j in range(0, len(cap_Q_values)):
        cap_name = cap_Q_list[j]
        cap_kvar = cap_Q_values[j]/1e3
        # cap_kvar = opfobj.capbk_df.iloc[j]['kvar']
        change_load_string1 = 'edit capacitor.' + str(cap_name) + ' kvar' + ' = ' + str(cap_kvar)
        logger.debug(f"{change_load_string1}")
        dss.run_command(change_load_string1)

    for j in range(0, len(pv_P_values)):
        pv_name = pv_P_list[j]
        pv_kw = pv_P_values[j]/1e3
        pv_kvar = pv_Q_values[j]/1e3
        # cap_kvar = opfobj.capbk_df.iloc[j]['kvar']
        change_load_string1 = 'edit PVsystem.' + str(pv_name) + ' kW' + ' = ' + str(pv_kw)
        change_load_string2 = 'edit PVsystem.' + str(pv_name) + ' kvar' + ' = ' + str(pv_kvar)

        logger.debug(f"{change_load_string1}")
        logger.debug(f"{change_load_string2}")
        dss.run_command(change_load_string1)
        dss.run_command(change_load_string2)
        # change taps format
        # Transformer.Reg1.wdg = 2
        # Tap = (0.00625  5 * 1 +)   ! Tap
        # 5
        # Transformer.Reg2.wdg = 2
        # Tap = (0.00625 5 * 1 +)    ! Tap
        # 5
        # Transformer.Reg3.wdg = 2
        # Tap = (0.00625  5 * 1 +)   ! Tap
        # 5
    for i in range(0, len(tap_vals_list)):
        tap_wdg_string = 'Transformer.' + str(tap_vals_list[i]) + '.wdg = ' + str(2)
        tap_number_string = 'Tap = (0.00625 ' + str(tap_vals_values[i]) + ' * 1 +)'
        logger.debug(f"{tap_wdg_string}")
        logger.debug(f"{tap_number_string}")
        dss.run_command(tap_wdg_string)
        dss.run_command(tap_number_string)

