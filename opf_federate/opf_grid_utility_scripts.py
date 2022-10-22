
from scipy.sparse import csc_matrix, linalg
import numpy as np
import os
import matplotlib.pyplot as plt
import json
import pyomo.environ as pyo
import random
import timeit

import logging
logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.DEBUG)

# plotting nodes and lines
v_node = [51, 54, 60, 18, 19, 20, 21, 24, 27]
line_node = [12, 13, 14, 15, 16, 17, 18, 19, 21]

#TODO: see if there is a better way other than to iterate through loops
def y_matrix_to_complex_array(matrix, rows, cols):
    array = np.zeros(shape=(rows, cols), dtype=complex)

    for k in range(0, rows):
        for j in range(0, cols):
            array[k, j] = matrix[k][j].real + 1j * matrix[k][j].imag
    return array

def matrix_to_array(matrix, rows, cols):
    array = np.zeros(shape=(rows, cols), dtype=float)

    for k in range(0, rows):
        for j in range(0, cols):
            array[k, j] = matrix[k][j]
    return array

def unpack_fdrvals_mats(opfobj):
    # opfobj.y_matrix = np.array(opfobj.topology.y_matrix)

    # opfobj.y_matrix = np.array(opfobj.topology)

    opfobj.y_matrix = y_matrix_to_complex_array(opfobj.y_matrix,
                                                opfobj.y_matrix.shape[0],
                                                opfobj.y_matrix.shape[1])

    # opfobj.y_line = np.array(opfobj.topology_flow)

    # opfobj.y_line = np.array(opfobj.topology_flow.y_matrix)

    opfobj.y_line = y_matrix_to_complex_array(opfobj.y_line,
                                              opfobj.y_line.shape[0],
                                              opfobj.y_line.shape[1])

    opfobj.Y_00 = csc_matrix(opfobj.y_matrix[0:3, 0:3])
    opfobj.Y_0L = csc_matrix(opfobj.y_matrix[0:3, 3:])
    opfobj.Y_L0 = csc_matrix(opfobj.y_matrix[3:, 0:3])
    opfobj.Y_LL = csc_matrix(opfobj.y_matrix[3:, 3:])

    opfobj.Yf_0 = csc_matrix(opfobj.y_line[:, 0:3])
    opfobj.Yf_L = csc_matrix(opfobj.y_line[:, 3:])

def unpack_fdrvals_vecs(opfobj):

    opfobj.voltages = np.array(opfobj.voltages_real) + 1j * np.array(opfobj.voltages_imag)
    opfobj.vL = opfobj.voltages[3:]
    opfobj.v0 = opfobj.voltages[0:3]
    opfobj.vL_abs_dss = np.abs(opfobj.vL)

    opfobj.i_f = opfobj.Yf_0.dot(opfobj.v0) + opfobj.Yf_L.dot(opfobj.vL)  # line flow from complex
    opfobj.i_f_abs = np.abs(opfobj.i_f)

    s_loss = np.conj(np.conj(opfobj.voltages.T).dot(opfobj.y_matrix).dot(opfobj.voltages))

    opfobj.p_loss_dss = np.real(s_loss)
    opfobj.q_loss_dss = -np.imag(s_loss)

    opfobj.flex_load_inc_matrix = matrix_to_array(opfobj.flex_info.adj_matrix,
                                                  len(opfobj.flex_info.adj_matrix),
                                                  len(opfobj.flex_info.adj_matrix[0]))
    opfobj.pv_inc_matrix = matrix_to_array(opfobj.pv_info.adj_matrix,
                                                  len(opfobj.pv_info.adj_matrix),
                                                  len(opfobj.pv_info.adj_matrix[0]))
    opfobj.cap_load_inc_matrix = matrix_to_array(opfobj.cap_info.adj_matrix,
                                                  len(opfobj.cap_info.adj_matrix),
                                                  len(opfobj.cap_info.adj_matrix[0]))

    opfobj.reg_inc_matrix = matrix_to_array(opfobj.tap_info.adj_matrix,
                                                  len(opfobj.tap_info.adj_matrix),
                                                  len(opfobj.tap_info.adj_matrix[0]))

    opfobj.pL = np.array(opfobj.powers_P)[3:]
    opfobj.qL = np.array(opfobj.powers_Q)[3:]
    opfobj.p_flex_load_init = opfobj.flex_load_inc_matrix.T.dot(opfobj.pL)
    opfobj.cap_bank_init = opfobj.cap_load_inc_matrix.T.dot(np.array(opfobj.cap_Q)[3:])
    opfobj.pv_p_init = opfobj.pv_inc_matrix.T.dot(np.array(opfobj.pv_P)[3:])
    opfobj.pv_q_init = opfobj.pv_inc_matrix.T.dot(np.array(opfobj.pv_Q)[3:])
    opfobj.reg_taps_init = np.array(opfobj.tap_vals)



def get_sens_matrices(opfobj):
    """
    linear sensitivity matrices of the form
    M_dx_dy, where the matrix entries (i,j) represent change in "x_ij (rows)" to change in y (columns) (dx_{i,j}/dy_{i,j})
    m_dx_dy where the array entries represent change in x to change in y (columns) dx/dy_{i,j}
    sY denotes the wye connected loads arranged as [pY;qY], i.e., for "n" nodes, size of pY is nx1, and sY is 2nx1
    So nodal sensitivity matrices are sized nx2n
    Line sensitivity matrices are sized lx2n, with "l" number of lines
    loss sensitivity vector is sized 1x2n
    """
    # Voltage sensitivity
    M_vL_sY = np.hstack((linalg.inv(opfobj.Y_LL).dot(np.linalg.inv(np.diag(np.conj(opfobj.vL)))),
                         -1j * linalg.inv(opfobj.Y_LL).dot(np.linalg.inv(np.diag(np.conj(opfobj.vL))))))

    K_vMag_sY = np.linalg.inv(np.diag(np.abs(opfobj.vL))).dot(np.real(np.diag(np.conj(opfobj.vL)).dot(M_vL_sY)))

    # losses sensitivities:
    # changes in active power loss to change in voltages
    # m_pl_y0l_v0 = np.conj(np.conj(opfobj.v0.T).dot(np.real(opfobj.Y_0L)))
    # m_pl_yl0_v0 = np.real(opfobj.Y_L0).dot(opfobj.v0)
    # m_pl_yll_vl = np.real(opfobj.Y_LL).dot(opfobj.vL.T)
    # m_pl_vL_yll = np.conj(np.conj(opfobj.vL.T).dot(np.real(opfobj.Y_LL)))
    # m_pl_vL_summ = m_pl_y0l_v0 + m_pl_yl0_v0 + m_pl_yll_vl + m_pl_vL_yll

    m_pl_y0l_v0 = np.conj(np.conj(opfobj.v0.T) @ (np.real(opfobj.Y_0L)))
    m_pl_yl0_v0 = np.real(opfobj.Y_L0) @ (opfobj.v0)
    m_pl_yll_vl = np.real(opfobj.Y_LL) @ (opfobj.vL.T)
    m_pl_vL_yll = np.conj(np.conj(opfobj.vL.T) @ (np.real(opfobj.Y_LL)))
    m_pl_vL_summ = m_pl_y0l_v0 + m_pl_yl0_v0 + m_pl_yll_vl + m_pl_vL_yll

    # collecting real and imaginary parts in manifold form -
    # for more info, see Bolognani et. al., "Fast power system analysis via implicit linearization of the power flow manifold"
    m_pl_vL = np.vstack((np.hstack((np.real(m_pl_vL_summ), np.imag(m_pl_vL_summ))),
                         np.hstack((np.imag(m_pl_vL_summ), -np.real(m_pl_vL_summ)))))
    # second rows are the imaginary parts of the loss functions, which can be proven to be zero!
    # change in active power loss to change in injection
    m_pl_sY = m_pl_vL.dot(np.vstack((np.real(M_vL_sY), np.imag(M_vL_sY))))[0]
    # # changes in reactive power loss to change in voltages
    # m_ql_y0l_v0 = np.conj(np.conj(opfobj.v0.T).dot(np.imag(-opfobj.Y_0L)))
    # # m_ql_y0l_v0 = np.conj(np.conj(opfobj.v0.T)@(np.imag(-opfobj.Y_0L)))
    # m_ql_yl0_v0 = np.imag(-opfobj.Y_L0).dot(opfobj.v0)
    # m_ql_yll_vl = np.imag(-opfobj.Y_LL).dot(opfobj.vL.T)
    # m_ql_vL_yll = np.conj(np.conj(opfobj.vL.T).dot(np.imag(-opfobj.Y_LL)))
    # m_ql_vL_summ = m_ql_y0l_v0 + m_ql_yl0_v0 + m_ql_yll_vl + m_ql_vL_yll
    # changes in reactive power loss to change in voltages
    m_ql_y0l_v0 = np.conj(np.conj(opfobj.v0.T) @ (np.imag(-opfobj.Y_0L)))
    # m_ql_y0l_v0 = np.conj(np.conj(opfobj.v0.T)@(np.imag(-opfobj.Y_0L)))
    m_ql_yl0_v0 = np.imag(-opfobj.Y_L0) @ (opfobj.v0)
    m_ql_yll_vl = np.imag(-opfobj.Y_LL) @ (opfobj.vL.T)
    m_ql_vL_yll = np.conj(np.conj(opfobj.vL.T) @ (np.imag(-opfobj.Y_LL)))
    m_ql_vL_summ = m_ql_y0l_v0 + m_ql_yl0_v0 + m_ql_yll_vl + m_ql_vL_yll
    # collecting real and imaginary parts in manifold form -
    # for more info, see Bolognani et. al., "Fast power system analysis via implicit linearization of the power flow manifold"
    m_ql_vL = np.vstack((np.hstack((np.real(m_ql_vL_summ), np.imag(m_ql_vL_summ))),
                         np.hstack((np.imag(m_ql_vL_summ), -np.real(m_ql_vL_summ)))))
    # second rows are the imaginary parts of the loss functions, which can be proven to be zero!
    # change in reactive power loss to change in injection
    m_ql_sY = m_ql_vL.dot(np.vstack((np.real(M_vL_sY), np.imag(M_vL_sY))))[0]

    # change of line flow "from nodes" to change in voltage
    M_if_vL = opfobj.Yf_L.dot(M_vL_sY)

    try:
        M_if_sY = np.linalg.inv(np.diag(np.conj(opfobj.i_f_abs))).dot(
            np.real(np.diag(np.conj(opfobj.i_f)).dot(M_if_vL)))
    except:
        logger.debug(
            f"there were few line flows zero, which means there is a very small voltage difference across certain nodes - adding a small constant to avoid that ")
        opfobj.i_f_abs[np.where(opfobj.i_f_abs == 0.0)[0]] = 1e-3
        M_if_sY = np.linalg.inv(np.diag(np.conj(opfobj.i_f_abs))).dot(
            np.real(np.diag(np.conj(opfobj.i_f)).dot(M_if_vL)))

    # separating variables in active and reactive powers
    opfobj.K_vMag_pY = K_vMag_sY[:, 0:len(opfobj.vL)]
    opfobj.K_vMag_qY = K_vMag_sY[:, len(opfobj.vL):]

    opfobj.M_if_pY = M_if_sY[:, 0:len(opfobj.vL)]
    opfobj.M_if_qY = M_if_sY[:, len(opfobj.vL):]
    opfobj.m_pl_pY = m_pl_sY[0:len(opfobj.vL)]
    opfobj.m_pl_qY = m_pl_sY[len(opfobj.vL):]
    opfobj.m_ql_pY = m_ql_sY[0:len(opfobj.vL)]
    opfobj.m_ql_qY = m_ql_sY[len(opfobj.vL):]
    
    

def test_lin_model(opfobj, load_change_test,
               opf_cap_lin_model_test,
               opf_xfrmr_lin_model_test,
               opf_flex_load_lin_model_test,
               opf_lin_model_all_vars_test,
               plotfig, savefig):
    """
    linear model of the grid utilizing the sensitivities from the above
    This model can be used to test the accuracy of linear model with the actual loading
    """
    # Voltage model
    # verify the multiplication rules and then the position of matrices

    # linear model is of the following type
    # voltage_cmplx_lin = w + M_v_p * p + M_v_q * q\

    # voltage_mag_lin = a + M_|v|_p * p + M_|v|_q * q (for global approximation)
    # voltage_mag_lin = v_hat_l + M_|v|_p * dp + M_|v|_q * dq
    # current_mag_lin = c + M_|i|_p * p + M_|i|_q * q
    # p_loss_lin = d + M_pl_p * p + M_pl_q * q
    # q_loss_lin = e + M_ql_p * p + M_ql_q * q
    # initialize coefficients
    # complex voltage coefficient
    w = -(linalg.inv(opfobj.Y_LL).dot(opfobj.Y_L0)).dot(opfobj.v0)
    # voltage magnitude coefficient (if you want to center around rated load
    # b = np.abs(opfobj.vL) \
    #     - opfobj.K_vMag_pY.dot(opfobj.pL) \
    #     - opfobj.K_vMag_qY.dot(opfobj.qL)
    # voltage magnitude coefficient (if you want to have a fixed-point linearization)
    b = abs(w)
    c = np.abs(opfobj.Yf_0.dot(opfobj.v0) + opfobj.Yf_L.dot(w))

    v_noload = np.hstack((opfobj.v0, w))
    # this is no load active power loss
    d = np.real(np.conj(np.conj(v_noload.T).dot(opfobj.y_matrix.toarray()).dot(v_noload)))
    # this is no load reactive power loss
    e = np.imag(np.conj(np.conj(v_noload.T).dot(opfobj.y_matrix.toarray()).dot(v_noload)))

    plt_available = 0

    if load_change_test:
        plt_available = 1
        test = 'flex_load_test'

        mult_range = np.arange(-0.75, 2.5, 0.1)

        # initialize linear variable vectors
        volt_mag_lin_range = np.zeros(shape=(len(opfobj.vL), len(mult_range)))
        current_lin_range = np.zeros(shape=(len(opfobj.i_f_abs), len(mult_range)))
        p_loss_lin_range = np.zeros(shape=(len(mult_range)))
        q_loss_lin_range = np.zeros(shape=(len(mult_range)))

        # initialize nonlinear variable vectors
        volt_mag_nonlin_range = np.zeros(shape=(len(opfobj.vL), len(mult_range)))
        current_nonlin_range = np.zeros(shape=(len(opfobj.i_f_abs), len(mult_range)))  # absolute flows complex
        p_loss_nonlin_range = np.zeros(shape=(len(mult_range)))
        q_loss_nonlin_range = np.zeros(shape=(len(mult_range)))
        for k in range(0, len(mult_range)):
            volt_mag_lin_range[:, k] = b + \
                                       opfobj.K_vMag_pY.dot(-opfobj.pL * mult_range[k]) + \
                                       opfobj.K_vMag_qY.dot(-opfobj.qL * mult_range[k])

            current_lin_range[:, k] = c + \
                                      opfobj.M_if_pY.dot(-opfobj.pL * mult_range[k]) + \
                                      opfobj.M_if_qY.dot(-opfobj.qL * mult_range[k])

            p_loss_lin_range[k] = d + \
                                  opfobj.m_pl_pY.dot(-opfobj.pL * mult_range[k]) + \
                                  opfobj.m_pl_qY.dot(-opfobj.qL * mult_range[k])

            q_loss_lin_range[k] = e + \
                                  opfobj.m_ql_pY.dot(-opfobj.pL * mult_range[k]) + \
                                  opfobj.m_ql_qY.dot(-opfobj.qL * mult_range[k])

            for j in range(0, len(opfobj.loads_df)):
                load_name = opfobj.loads_df.iloc[j]['Name']
                load_kW = opfobj.loads_df.iloc[j]['kW']
                load_kvar = opfobj.loads_df.iloc[j]['kvar']
                change_load_string1 = 'edit load.' + str(load_name) + ' kW' + ' = ' + str(load_kW * mult_range[k])
                change_load_string2 = 'edit load.' + str(load_name) + ' kvar' + ' = ' + str(load_kvar * mult_range[k])
                logger.debug(f"{change_load_string1}")
                logger.debug(f"{change_load_string2}")
                opfobj.dss_obj.run_command(change_load_string1)
                opfobj.dss_obj.run_command(change_load_string2)

            dss_funcs.snapshot_run(opfobj.dss_obj)
            opfobj.voltages, opfobj.v0, opfobj.vL = dss_funcs.get_voltage_vector(opfobj.dss_obj)
            # opfobj.snapshot_run()
            # opfobj.y_ordered_voltage_array()

            i_f = opfobj.Yf_0.dot(opfobj.v0) + opfobj.Yf_L.dot(opfobj.vL)  # line flow from complex
            volt_mag_nonlin_range[:, k] = np.abs(opfobj.vL)
            current_nonlin_range[:, k] = np.abs(i_f)  # absolute flows complex
            p_loss_nonlin_range[k] = opfobj.dss_obj.Circuit.Losses()[0]
            q_loss_nonlin_range[k] = opfobj.dss_obj.Circuit.Losses()[1]

        logger.debug("==Linear Model Testing for flex load change and its response complete====")

    if opf_cap_lin_model_test:
        plt_available = 1

        test = '_cap_test'
        mult_range = np.arange(0, 1, 0.05)

        # the multiply range shows the activation of capacitor in steps of 0.05 (proxy for activating cap bank numbers)
        # , i.e., as multiplier increase from 0 to 1, we are injecting reactive power from the capacitor
        # hence the following equation
        # v_linear = v_operating_point + M_q * flex_cap
        # shows the injection in capacitor supplied reactive power and the voltage will rise. Same could be said about losses which will be reduced and line flows
        # Reactors will be the other way, i.e., the sign convention will be from -1 to 0.

        # initialize linear variable vectors
        volt_mag_lin_range = np.zeros(shape=(len(opfobj.vL), len(mult_range)))
        current_lin_range = np.zeros(shape=(len(opfobj.i_f_abs), len(mult_range)))
        p_loss_lin_range = np.zeros(shape=(len(mult_range)))
        q_loss_lin_range = np.zeros(shape=(len(mult_range)))

        # initialize nonlinear variable vectors
        volt_mag_nonlin_range = np.zeros(shape=(len(opfobj.vL), len(mult_range)))
        current_nonlin_range = np.zeros(shape=(len(opfobj.i_f_abs), len(mult_range)))  # absolute flows complex
        p_loss_nonlin_range = np.zeros(shape=(len(mult_range)))
        q_loss_nonlin_range = np.zeros(shape=(len(mult_range)))
        conv_factor = 1000  # to convert from kW to Watt
        rated_cap = opfobj.cap_load_inc_matrix.dot(opfobj.capbk_df['kvar'].values) * conv_factor
        for k in range(0, len(mult_range)):
            volt_mag_lin_range[:, k] = opfobj.vL_abs_dss + \
                                       opfobj.K_vMag_qY.dot(rated_cap * mult_range[
                                           k]) / 3  # with normal linearization it was giving overestimation - so divided it by 3

            current_lin_range[:, k] = opfobj.i_f_abs + \
                                      opfobj.M_if_qY.dot(rated_cap * mult_range[k]) / 3

            p_loss_lin_range[k] = opfobj.p_loss_dss + \
                                  opfobj.m_pl_qY.dot(rated_cap * mult_range[k]) / 3

            q_loss_lin_range[k] = opfobj.q_loss_dss + \
                                  opfobj.m_ql_qY.dot(rated_cap * mult_range[k]) / 3

            cap_vector = opfobj.cap_load_inc_matrix.T.dot(rated_cap * (1 + mult_range[k])) / 1000
            for j in range(0, len(opfobj.capbk_df)):
                cap_name = opfobj.capbk_df.iloc[j]['Name']
                cap_kvar = cap_vector[j]
                # cap_kvar = opfobj.capbk_df.iloc[j]['kvar']
                change_load_string1 = 'edit capacitor.' + str(cap_name) + ' kvar' + ' = ' + str(cap_kvar)
                logger.debug(f"{change_load_string1}")
                opfobj.dss_obj.run_command(change_load_string1)

            dss_funcs.snapshot_run(opfobj.dss_obj)
            opfobj.voltages, opfobj.v0, opfobj.vL = dss_funcs.get_voltage_vector(opfobj.dss_obj)
            # opfobj.y_ordered_voltage_array()

            i_f = opfobj.Yf_0.dot(opfobj.v0) + opfobj.Yf_L.dot(opfobj.vL)  # line flow from complex
            volt_mag_nonlin_range[:, k] = np.abs(opfobj.vL)
            current_nonlin_range[:, k] = np.abs(i_f)  # absolute flows complex
            p_loss_nonlin_range[k] = opfobj.dss_obj.Circuit.Losses()[0]
            q_loss_nonlin_range[k] = opfobj.dss_obj.Circuit.Losses()[1]

    if opf_xfrmr_lin_model_test:
        plt_available = 1

        test = 'xfrmr_reg_test'
        mult_range = np.arange(-16, 16, 1)
        # representing minimum and maximum tap numbers
        # transformer tap model
        # xmer_voltage_expr1 = tau*diag(v)* C_r
        # tau is the regulator step change ratio .00625
        # v is the operating point voltage
        # C is the matrix relating node voltage to upstream nodes
        check = 1
        # initialize linear variable vectors
        volt_mag_lin_range = np.zeros(shape=(len(opfobj.vL), len(mult_range)))
        current_lin_range = np.zeros(shape=(len(opfobj.i_f_abs), len(mult_range)))
        p_loss_lin_range = np.zeros(shape=(len(mult_range)))
        q_loss_lin_range = np.zeros(shape=(len(mult_range)))

        # initialize nonlinear variable vectors
        volt_mag_nonlin_range = np.zeros(shape=(len(opfobj.vL), len(mult_range)))
        current_nonlin_range = np.zeros(shape=(len(opfobj.i_f_abs), len(mult_range)))  # absolute flows complex
        p_loss_nonlin_range = np.zeros(shape=(len(mult_range)))
        q_loss_nonlin_range = np.zeros(shape=(len(mult_range)))

        tau = 0.00625
        # get nominal taps
        # reg_taps = []
        # reg_names = opfobj.dss_obj.RegControls.AllNames()
        # for i in range(0, len(reg_names)):
        #     opfobj.dss_obj.Circuit.SetActiveClass('RegControl')
        #     opfobj.dss_obj.Circuit.SetActiveElement(reg_names[i])
        #     reg_taps.append(opfobj.dss_obj.RegControls.TapNumber())
        # reg_taps = np.array(reg_taps)
        for k in range(0, len(mult_range)):

            volt_mag_lin_range[:, k] = opfobj.vL_abs_dss + tau * (np.diag(opfobj.vL_abs_dss).dot(opfobj.reg_inc_matrix)).dot(
                opfobj.reg_taps_init - mult_range[k])

            current_lin_range[:, k] = opfobj.i_f_abs
            p_loss_lin_range[k] = opfobj.p_loss_dss
            q_loss_lin_range[k] = opfobj.q_loss_dss

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
            for i in range(0, len(opfobj.reg_order_name)):
                opfobj.dss_obj.Circuit.SetActiveClass('RegControl')
                opfobj.dss_obj.Circuit.SetActiveElement(opfobj.reg_order_name[i])
                tap_wdg_string = 'Transformer.' + str(opfobj.reg_order_name[i]) + '.wdg = ' + str(
                    opfobj.dss_obj.RegControls.TapWinding())
                tap_number_string = 'Tap = (0.00625 ' + str(opfobj.reg_taps_init[i] - mult_range[k]) + ' * 1 +)'
                logger.debug(f"{tap_wdg_string}")
                logger.debug(f"{tap_number_string}")
                opfobj.dss_obj.run_command(tap_wdg_string)
                opfobj.dss_obj.run_command(tap_number_string)

            # opfobj.snapshot_run()
            # opfobj.y_ordered_voltage_array()

            dss_funcs.snapshot_run(opfobj.dss_obj)
            opfobj.voltages, opfobj.v0, opfobj.vL = dss_funcs.get_voltage_vector(opfobj.dss_obj)

            i_f = opfobj.Yf_0.dot(opfobj.v0) + opfobj.Yf_L.dot(opfobj.vL)  # line flow from complex
            volt_mag_nonlin_range[:, k] = np.abs(opfobj.vL)
            current_nonlin_range[:, k] = np.abs(i_f)  # absolute flows complex
            p_loss_nonlin_range[k] = opfobj.dss_obj.Circuit.Losses()[0]
            q_loss_nonlin_range[k] = opfobj.dss_obj.Circuit.Losses()[1]

    if opf_flex_load_lin_model_test:
        plt_available = 1

        test = '_only_flex_load_test'
        # u_init = np.array([800.0, 800.0, 800.0, 160.0, 120.0, 120.0, 170.0, 485.0, 68.0, 290.0, 17.0, 66.0, 117.0])
        # u_idx = ['671a', '671b', '671c', '634a', '634b', '634c', '645', '675a', '675b', '675c', '670a', '670b',
        #          '670c']
        conv_factor = 1000
        mult_range = np.arange(0, 1, 0.05)
        # the multiply range shows the convention of load sheding, i.e., as flex load multiplier increase from 0 to 1, it is decreasing the load at the node
        # hence the following equation
        # v_linear = v_operating_point + M_p * flex_load
        # shows the reduction in load and its impact on the voltage profile. Same could be said about losses and line flows
        # for DGs it will become the other way, i.e., the sign convention will be from -1 to 0.
        # volt_mag_lin_range[:, k] = opfobj.vL_abs_dss + opfobj.K_vMag_pY.dot(flex_load * mult_range[k])

        # initialize linear variable vectors
        volt_mag_lin_range = np.zeros(shape=(len(opfobj.vL), len(mult_range)))
        current_lin_range = np.zeros(shape=(len(opfobj.i_f_abs), len(mult_range)))
        p_loss_lin_range = np.zeros(shape=(len(mult_range)))
        q_loss_lin_range = np.zeros(shape=(len(mult_range)))

        # initialize nonlinear variable vectors
        volt_mag_nonlin_range = np.zeros(shape=(len(opfobj.vL), len(mult_range)))
        current_nonlin_range = np.zeros(shape=(len(opfobj.i_f_abs), len(mult_range)))  # absolute flows complex
        p_loss_nonlin_range = np.zeros(shape=(len(mult_range)))
        q_loss_nonlin_range = np.zeros(shape=(len(mult_range)))

        # flexible load vector in the form of grid nodes
        flex_load = opfobj.flex_load_inc_matrix.dot(opfobj.p_flex_load_init)

        for k in range(0, len(mult_range)):
            test = 'flex_load_opf_test'

            volt_mag_lin_range[:, k] = opfobj.vL_abs_dss + opfobj.K_vMag_pY.dot(flex_load * mult_range[k])

            current_lin_range[:, k] = opfobj.i_f_abs + opfobj.M_if_pY.dot(flex_load * mult_range[k]) * 1.5

            p_loss_lin_range[k] = opfobj.p_loss_dss + opfobj.m_pl_pY.dot(flex_load * mult_range[k]) / 2

            q_loss_lin_range[k] = opfobj.q_loss_dss + opfobj.m_ql_pY.dot(flex_load * mult_range[k]) / 2

            load_vector = opfobj.flex_load_inc_matrix.T.dot(opfobj.pL - flex_load * mult_range[k]) / 1000
            for j in range(0, len(opfobj.u_idx)):
                load_name = opfobj.u_idx[j]
                load_kW = load_vector[j]
                change_load_string1 = 'edit load.' + str(load_name) + ' kW' + ' = ' + str(load_kW)
                logger.debug(f"{change_load_string1}")
                opfobj.dss_obj.run_command(change_load_string1)

            # opfobj.snapshot_run()
            # opfobj.y_ordered_voltage_array()
            
            dss_funcs.snapshot_run(opfobj.dss_obj)
            opfobj.voltages, opfobj.v0, opfobj.vL = dss_funcs.get_voltage_vector(opfobj.dss_obj)

            i_f = opfobj.Yf_0.dot(opfobj.v0) + opfobj.Yf_L.dot(opfobj.vL)  # line flow from complex
            volt_mag_nonlin_range[:, k] = np.abs(opfobj.vL)
            current_nonlin_range[:, k] = np.abs(i_f)  # absolute flows complex
            p_loss_nonlin_range[k] = opfobj.dss_obj.Circuit.Losses()[0]
            q_loss_nonlin_range[k] = opfobj.dss_obj.Circuit.Losses()[1]

    if opf_lin_model_all_vars_test:
        plt_available = 1

        test = '_all_var_test'
        # u_init = np.array([800.0, 800.0, 800.0, 160.0, 120.0, 120.0, 170.0, 485.0, 68.0, 290.0, 17.0, 66.0, 117.0])
        # u_idx = ['671a', '671b', '671c', '634a', '634b', '634c', '645', '675a', '675b', '675c', '670a', '670b',
        #          '670c']
        conv_factor = 1000

        mult_range = np.arange(0, 1, 0.05)
        tap_mult_range = -mult_range * 16
        volt_mag_lin_range = np.zeros(shape=(len(opfobj.vL), len(mult_range)))
        current_lin_range = np.zeros(shape=(len(opfobj.i_f_abs), len(mult_range)))
        p_loss_lin_range = np.zeros(shape=(len(mult_range)))
        q_loss_lin_range = np.zeros(shape=(len(mult_range)))

        # initialize nonlinear variable vectors
        volt_mag_nonlin_range = np.zeros(shape=(len(opfobj.vL), len(mult_range)))
        current_nonlin_range = np.zeros(shape=(len(opfobj.i_f_abs), len(mult_range)))  # absolute flows complex
        p_loss_nonlin_range = np.zeros(shape=(len(mult_range)))
        q_loss_nonlin_range = np.zeros(shape=(len(mult_range)))

        # flexible load vector in the form of grid nodes
        flex_load = opfobj.flex_load_inc_matrix.dot(opfobj.p_flex_load_init)
        # cap bank vector
        rated_cap = opfobj.cap_load_inc_matrix.dot(opfobj.capbk_df['kvar'].values) * conv_factor
        # pv system p injection
        pv_p = opfobj.pv_inc_matrix.dot(opfobj.pv_p_init)
        # pv system q injection
        pv_q = opfobj.pv_inc_matrix.dot(opfobj.pv_q_init)
        pv_s = opfobj.pv_inc_matrix.dot(opfobj.pv_s_init)
        pv_s_vector = opfobj.pv_inc_matrix.T.dot(pv_s) / 1000
        tau = 0.00625

        for k in range(0, len(mult_range)):
            test = 'all_var_test'
            pv_p_mult = pv_p * mult_range[k]
            try:
                q_2 = pv_s ** 2 - pv_p_mult ** 2
                pv_q_mult = np.sqrt(q_2.astype(float))*0
            except:
                pv_q_mult = np.zeros(shape=(len(opfobj.vL)))*0
            volt_mag_lin_range[:, k] = opfobj.vL_abs_dss \
                                       + opfobj.K_vMag_pY.dot( (flex_load+pv_p_mult) * mult_range[k])/2 \
                                       + opfobj.K_vMag_qY.dot( (rated_cap+pv_q_mult) * mult_range[k])/2  \
                                       + tau * (np.diag(opfobj.vL_abs_dss).dot(opfobj.reg_inc_matrix)).dot(
                opfobj.reg_taps_init - tap_mult_range[k])

            current_lin_range[:, k] = opfobj.i_f_abs \
                                      + opfobj.M_if_pY.dot((flex_load+pv_p_mult) * mult_range[k])/2  \
                                      + opfobj.M_if_qY.dot((rated_cap+pv_q_mult) * mult_range[k])/2

            p_loss_lin_range[k] = opfobj.p_loss_dss \
                                  + opfobj.m_pl_pY.dot((flex_load+pv_p_mult) * mult_range[k])/5 \
                                  + opfobj.m_pl_qY.dot((rated_cap+pv_q_mult) * mult_range[k])/5

            q_loss_lin_range[k] = opfobj.q_loss_dss \
                                  + opfobj.m_ql_pY.dot((flex_load+pv_p_mult) * mult_range[k])/3 + \
                                  opfobj.m_ql_qY.dot((rated_cap+pv_q_mult) * mult_range[k])/5

            load_vector = opfobj.flex_load_inc_matrix.T.dot(opfobj.pL - flex_load * mult_range[k]) / 1000
            for j in range(0, len(opfobj.u_idx)):
                load_name = opfobj.u_idx[j]
                load_kW = load_vector[j]
                change_load_string1 = 'edit load.' + str(load_name) + ' kW' + ' = ' + str(load_kW)
                logger.debug(f"{change_load_string1}")
                opfobj.dss_obj.run_command(change_load_string1)

            cap_vector = opfobj.cap_load_inc_matrix.T.dot(rated_cap * (1 + mult_range[k])) / 1000
            for j in range(0, len(opfobj.capbk_df)):
                cap_name = opfobj.capbk_df.iloc[j]['Name']
                cap_kvar = cap_vector[j]
                # cap_kvar = opfobj.capbk_df.iloc[j]['kvar']
                change_load_string1 = 'edit capacitor.' + str(cap_name) + ' kvar' + ' = ' + str(cap_kvar)
                logger.debug(f"{change_load_string1}")
                opfobj.dss_obj.run_command(change_load_string1)

            pv_p_vector = opfobj.pv_inc_matrix.T.dot(pv_p - pv_p_mult) / 1000
            try:
                q_2 = pv_s_vector ** 2 - pv_p_vector ** 2
                pv_q_vector = np.sqrt(q_2.astype(float))*0
            except:
                pv_q_vector = np.zeros(shape=(len(opfobj.vL)))*0
            # pv_q_vector = -opfobj.pv_inc_matrix.T.dot(pv_q - pv_q_mult) / 1000
            for j in range(0, len(opfobj.pvs_df)):
                pv_name = opfobj.pvs_df.iloc[j]['Name']
                pv_kw = pv_p_vector[j]
                pv_kvar = pv_q_vector[j]
                # cap_kvar = opfobj.capbk_df.iloc[j]['kvar']
                change_load_string1 = 'edit PVsystem.' + str(pv_name) + ' kW' + ' = ' + str(pv_kw)
                change_load_string2 = 'edit PVsystem.' + str(pv_name) + ' kvar' + ' = ' + str(pv_kvar)

                logger.debug(f"{change_load_string1}")
                logger.debug(f"{change_load_string2}")
                opfobj.dss_obj.run_command(change_load_string1)
                opfobj.dss_obj.run_command(change_load_string2)
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
            for i in range(0, len(opfobj.reg_order_name)):
                # opfobj.dss_obj.Circuit.SetActiveClass('RegControl')
                # opfobj.dss_obj.Circuit.SetActiveElement(opfobj.reg_order_name[i])
                # tap_wdg_string = 'Transformer.' + str(opfobj.reg_order_name[i]) + '.wdg = ' + str(
                #     opfobj.dss_obj.RegControls.TapWinding())
                tap_wdg_string = 'Transformer.' + str(opfobj.reg_order_name[i]) + '.wdg = ' + str(2)
                tap_number_string = 'Tap = (0.00625 ' + str(opfobj.reg_taps_init[i] - tap_mult_range[k]) + ' * 1 +)'
                logger.debug(f"{tap_wdg_string}")
                logger.debug(f"{tap_number_string}")
                opfobj.dss_obj.run_command(tap_wdg_string)
                opfobj.dss_obj.run_command(tap_number_string)

            # opfobj.snapshot_run()
            # opfobj.y_ordered_voltage_array()
            
            dss_funcs.snapshot_run(opfobj.dss_obj)
            opfobj.voltages, opfobj.v0, opfobj.vL = dss_funcs.get_voltage_vector(opfobj.dss_obj)

            i_f = opfobj.Yf_0.dot(opfobj.v0) + opfobj.Yf_L.dot(opfobj.vL)  # line flow from complex
            volt_mag_nonlin_range[:, k] = np.abs(opfobj.vL)
            current_nonlin_range[:, k] = np.abs(i_f)  # absolute flows complex
            p_loss_nonlin_range[k] = opfobj.dss_obj.Circuit.Losses()[0]
            q_loss_nonlin_range[k] = opfobj.dss_obj.Circuit.Losses()[1]

    if (plotfig is True) and (plt_available == 1):

        if savefig is True:
            dir_path = os.getcwd() + r'/../../../plots/'
            # dir_path = os.getcwd()+'\\..\\..\\..\\plots\\'
            logger.debug(f"plotting path {dir_path}")
            check_folder = os.path.isdir(dir_path)
            if not check_folder:
                os.makedirs(dir_path)
            logger.debug(f"plotting path: {dir_path}")
            voltage_plot_path = dir_path + 'v_plot_node' + str(v_node[0]) + '_' + test
            flow_plot_path = dir_path + 'flow_plot_node' + str(line_node[0]) + '_' + test
            loss_plot_path = dir_path + 'losses_plot' + '_' + test
            # vnode order ['SOURCEBUS.1', 'SOURCEBUS.2', 'SOURCEBUS.3', '650.1', '650.2', '650.3', ...
            # 'RG60.1', 'RG60.2', 'RG60.3', '633.1', '633.2', '633.3', '634.1', '634.2', '634.3', ...
            # '632.1', '632.2', '632.3', '670.1', '670.2', '670.3', '671.1', '671.2', '671.3', ...
            # '680.1', '680.2', '680.3', '645.3', '645.2', '646.3', '646.2', '692.1', '692.2', '692.3', '675.1', '675.2', '675.3', '684.1', '684.3', '611.3', '652.1']

        # voltage plots
        fig, ax = plt.subplots(len(v_node), 1, sharex=True, figsize=(16, 13))
        for i in range(0, len(v_node)):
            ax[i].plot(mult_range, volt_mag_nonlin_range[v_node[i], :], 'r^', label='OpenDSS')
            ax[i].plot(mult_range, volt_mag_lin_range[v_node[i], :], 'b--', label='Linear')
            if i == len(v_node)-1:
                ax[i].set(xlabel='load multiplier', ylabel='Volts',
                       title='Node.'+opfobj.node_order[v_node[i]+3])
            else:
                ax[i].set(ylabel='Volts',
                       title='Node.'+opfobj.node_order[v_node[i]+3])
            # Show a legend
            ax[i].legend()
            ax[i].grid()
        if savefig is True:
            fig.savefig(voltage_plot_path, bbox_inches='tight')

        # line flow plots
        fig, ax = plt.subplots(len(line_node), 1, sharex=True, figsize=(16, 13))
        for i in range(0, len(line_node)):
            ax[i].plot(mult_range, current_nonlin_range[line_node[i], :], 'r^', label='OpenDSS')
            ax[i].plot(mult_range, current_lin_range[line_node[i], :], 'b--', label='Linear')
            if i == len(line_node)-1:
                ax[i].set(xlabel='load multiplier', ylabel='Amps',
                       title='Line'+str(line_node[i]))
            else:
                ax[i].set(ylabel='Amps',
                       title='Line'+str(line_node[i]))
            # Show a legend
            ax[i].legend()
            ax[i].grid()

        if savefig is True:
            fig.savefig(flow_plot_path, bbox_inches='tight')


        # loss plots
        fig, ax = plt.subplots(2,1,sharex=True)
        ax[0].plot(mult_range, p_loss_nonlin_range, 'r^', label='OpenDSS')
        ax[0].plot(mult_range, p_loss_lin_range, 'b--', label='Linear')
        ax[0].set(xlabel='load multiplier', ylabel='Watts',
                   title='Active Power Losses')
        ax[0].legend()
        ax[0].grid()
        ax[1].plot(mult_range, q_loss_nonlin_range, 'r^', label='OpenDSS')
        ax[1].plot(mult_range, q_loss_lin_range, 'b--', label='Linear')
        ax[1].set(xlabel='load multiplier', ylabel='Vars',
                  title='Reactive Power Losses')
        # Show a legend
        ax[1].legend()
        ax[1].grid()

        if savefig is True:
            fig.savefig(loss_plot_path, bbox_inches='tight')

        plt.show()


def solve_central_optimization(opfobj):

    with open("flex_load_info.json") as f:
        flex_load_info = json.load(f)
    u_idx = flex_load_info["flex_node_idx"]
    u_init = flex_load_info["flex_node_init"]
    u_unit = flex_load_info["units"]

    cost_a = np.array(u_init) / 100
    cost_b = np.array(u_init) / 1000
    cost_p_gen_a = np.max(cost_a) * 100  # promote local production/loadshed
    cost_p_gen_b = np.mean(cost_b) * 0.1

    # cost_bank_a = np.array([200, 200, 200, 100])/100
    cost_bank_b = opfobj.cap_bank_init / 1000
    cost_q_gen_a = np.mean(cost_bank_b) * 100  # promote local production/capbank activation
    cost_q_gen_b = np.mean(cost_bank_b) * 0.1
    cost_reg_b = np.mean(cost_b) * 10 * np.ones(len(opfobj.reg_taps_init))
    cost_pv_p_a = opfobj.pv_p_init / 1e6
    cost_pv_p_b = opfobj.pv_q_init / 1e7
    cost_pv_q_b = 0.1 + opfobj.pv_q_init / 1e6


    def pflex_max_def(m, n):
        return m.p_flex_load_var[n] <= p_flex_load_var_limit_max[n]


    def pflex_min_def(m, n):
        return m.p_flex_load_var[n] >= p_flex_load_var_limit_min[n]


    def pv_p_max_def(m, n):
        return m.p_pv[n] <= pv_p_max[n]


    def pv_p_min_def(m, n):
        return m.p_pv[n] >= pv_p_min[n]


    def pv_q_max_def(m, n):
        return m.q_pv[n] <= pv_q_max[n]


    def pv_q_min_def(m, n):
        return m.q_pv[n] >= pv_q_min[n]


    def cap_max_def(m, n):
        return m.cap_active_number[n] <= cap_power_max[n]


    def cap_min_def(m, n):
        return m.cap_active_number[n] >= cap_power_min[n]


    def xmer_min_def(m, n):
        return m.xmer_tap_number[n] >= xmer_min_taps[n]


    def xmer_max_def(m, n):
        return m.xmer_tap_number[n] <= xmer_max_taps[n]


    def volt_max_def(m, n):
        return m.voltage_approx[n] <= voltage_max_lims[n]


    def volt_min_def(m, n):
        return m.voltage_approx[n] >= voltage_min_lims[n]


    def line_flow_lim_max_def(m, l):
        return m.line_flow_approx[l] <= line_ratings[l]


    def line_flow_lim_min_def(m, l):
        return m.line_flow_approx[l] >= -line_ratings[l]

        # linear equations of the form are of the form

    def power_balance_def_P(m):
        p_loss_p_flex_matrix = opfobj.m_pl_pY.dot(opfobj.flex_load_inc_matrix)
        p_loss_p_pv_matrix = opfobj.m_pl_pY.dot(opfobj.pv_inc_matrix)
        p_loss_q_pv_matrix = opfobj.m_pl_qY.dot(opfobj.pv_inc_matrix)
        p_loss_cap_matrix = opfobj.m_pl_qY.dot(opfobj.cap_load_inc_matrix)

        p_loss_flex_expr = (p_loss_p_flex_matrix[j] * (m.p_flex_load_var[j]) for j in flex_node_idx)
        p_loss_p_pv_expr = (p_loss_p_pv_matrix[j] * (m.p_pv[j]) for j in pv_node_idx)
        p_loss_q_pv_expr = (p_loss_q_pv_matrix[j] * (m.q_pv[j]) for j in pv_node_idx)
        p_loss_cap_expr = (p_loss_cap_matrix[j] * (m.cap_active_number[j]) for j in cap_node_idx)
        p_fixed_load_sum = sum(opfobj.pL[j] for j in node_idx)
        p_flex_load_sum = sum(m.p_flex_load_var[j] for j in flex_node_idx)
        p_pv_sum = sum(m.p_pv[j] for j in pv_node_idx)
        return p_fixed_load_sum \
               - p_flex_load_sum - p_pv_sum \
               + sum(p_loss_cap_expr) + sum(p_loss_q_pv_expr) \
               + sum(p_loss_flex_expr) + sum(p_loss_p_pv_expr) \
               + opfobj.p_loss_dss == m.p_net_import


    def power_balance_def_Q(m):
        q_loss_p_flex_matrix = opfobj.m_ql_pY.dot(opfobj.flex_load_inc_matrix)
        q_loss_p_pv_matrix = opfobj.m_ql_pY.dot(opfobj.pv_inc_matrix)
        q_loss_q_pv_matrix = opfobj.m_ql_qY.dot(opfobj.pv_inc_matrix)
        q_loss_cap_matrix = opfobj.m_ql_qY.dot(opfobj.cap_load_inc_matrix)

        q_loss_flex_expr = (q_loss_p_flex_matrix[j] * (m.p_flex_load_var[j]) for j in flex_node_idx)
        q_loss_p_pv_expr = (q_loss_p_pv_matrix[j] * (m.p_pv[j]) for j in pv_node_idx)
        q_loss_q_pv_expr = (q_loss_q_pv_matrix[j] * (m.q_pv[j]) for j in pv_node_idx)
        q_loss_cap_expr = (q_loss_cap_matrix[j] * (m.cap_active_number[j]) for j in cap_node_idx)
        q_cap_flex_expr = sum(m.cap_active_number[j] for j in cap_node_idx)
        q_q_pv_expr = sum(m.q_pv[j] for j in pv_node_idx)
        q_fixed_load_expr = sum(opfobj.qL[j] for j in node_idx)
        return q_fixed_load_expr \
               - q_cap_flex_expr - q_q_pv_expr \
               + sum(q_loss_cap_expr) + sum(q_loss_q_pv_expr) \
               + sum(q_loss_flex_expr) + sum(q_loss_p_pv_expr) \
               + opfobj.q_loss_dss == m.q_net_import


    def voltage_approx_def(m, i):
        flex_voltage_matrix = opfobj.K_vMag_pY.dot(opfobj.flex_load_inc_matrix)
        cap_voltage_matrix = opfobj.K_vMag_qY.dot(opfobj.cap_load_inc_matrix)
        p_pv_voltage_matrix = opfobj.K_vMag_pY.dot(opfobj.pv_inc_matrix)
        q_pv_voltage_matrix = opfobj.K_vMag_qY.dot(opfobj.pv_inc_matrix)
        flex_voltage_expr = flex_voltage_matrix.dot(m.p_flex_load_var)
        cap_voltage_expr = cap_voltage_matrix.dot(m.cap_active_number)
        p_pv_voltage_expr = p_pv_voltage_matrix.dot(m.p_pv)
        q_pv_voltage_expr = q_pv_voltage_matrix.dot(m.q_pv)
        # xmer_voltage_temp = utils.reg_inc_matrix.dot(utils.reg_taps_init - m.xmer_tap_number)
        # xmer_voltage_expr = tau * np.diag(utils.vL_abs_dss).dot(xmer_voltage_temp)
        # try this
        xmer_voltage_expr = tau * (np.diag(opfobj.vL_abs_dss).dot(opfobj.reg_inc_matrix)).dot(
            opfobj.reg_taps_init - m.xmer_tap_number)
        return m.voltage_approx[i] \
               == opfobj.vL_abs_dss[i] \
               + cap_voltage_expr[i] + q_pv_voltage_expr[i] \
               + flex_voltage_expr[i] + p_pv_voltage_expr[i] \
               + xmer_voltage_expr[i]


    def line_flow_approx_def(m, l):
        flex_flow_matrix = opfobj.M_if_pY.dot(opfobj.flex_load_inc_matrix)
        cap_flow_matrix = opfobj.M_if_qY.dot(opfobj.cap_load_inc_matrix)
        p_pv_flow_matrix = opfobj.M_if_pY.dot(opfobj.pv_inc_matrix)
        q_pv_flow_matrix = opfobj.M_if_qY.dot(opfobj.pv_inc_matrix)
        flex_flow_expr = flex_flow_matrix.dot(m.p_flex_load_var)
        cap_flow_expr = cap_flow_matrix.dot(m.cap_active_number)
        p_pv_flow_expr = p_pv_flow_matrix.dot(m.p_pv)
        q_pv_flow_expr = q_pv_flow_matrix.dot(m.q_pv)
        return m.line_flow_approx[l] == opfobj.i_f_abs[l] + cap_flow_expr[l] + flex_flow_expr[l] \
               + p_pv_flow_expr[l] + q_pv_flow_expr[l]

    def obj_rule(m):
        logger.debug(f'cost p gen a {cost_p_gen_a}')
        logger.debug(f'cost p gen b {cost_p_gen_b}')
        logger.debug(f'cost a {cost_a}')
        logger.debug(f'cost b {cost_b}')
        logger.debug(f'cost_bank_b {cost_bank_b}')
        logger.debug(f'cost_q_gen_a {cost_q_gen_a}')
        logger.debug(f'cost_q_gen_b {cost_q_gen_b}')
        logger.debug(f'cost_reg_b {cost_reg_b}')
        logger.debug(f'cost_pv_p_b {cost_pv_p_a}')
        logger.debug(f'cost_pv_q_b {cost_pv_q_b}')

        logger.debug(f'flex_node_idx {flex_node_idx}')
        logger.debug(f'num_xmers_idx {num_xmers_idx}')
        logger.debug(f'cap_node_idx {cap_node_idx}')
        logger.debug(f'pv_node_idx {pv_node_idx}')

        # logger.debug(f'pflexvar {type(m.p_flex_load_var)}')
        # logger.debug(f'm.xmer_tap_number {type(m.xmer_tap_number)}')
        # logger.debug(f'm.cap_active_number {type(m.cap_active_number)}')
        # logger.debug(f'm.p_pv {type(m.p_pv)}')
        # logger.debug(f'm.p_net_import {type(m.p_net_import)}')
        # logger.debug(f'm.q_net_import {type(m.q_net_import)}')


        return sum((m.p_flex_load_var[i] - opfobj.p_flex_load_init[i] ) * cost_a[i]
                   + 0.5 * (m.p_flex_load_var[i] - opfobj.p_flex_load_init[i] ) * cost_b[i] * (
                               m.p_flex_load_var[i] - opfobj.p_flex_load_init[i] ) for i in flex_node_idx) \
               + 0.5 * sum((m.xmer_tap_number[j] - opfobj.reg_taps_init[j] ) * cost_reg_b[j] * (
                    m.xmer_tap_number[j] - opfobj.reg_taps_init[j] ) for j in num_xmers_idx) \
               + 0.5 * sum((m.cap_active_number[j] - opfobj.cap_bank_init[j]) * cost_bank_b[j] * (
                    m.cap_active_number[j] - opfobj.cap_bank_init[j]) for j in cap_node_idx) \
               + sum((m.p_pv[i] - opfobj.pv_p_init[i]) * cost_pv_p_a[i]
                     + 0.5 * (m.p_pv[i] - opfobj.pv_p_init[i]) * cost_pv_p_b[i] * (m.p_pv[i] - opfobj.pv_p_init[i]) for
                     i in pv_node_idx) \
               + 0.5 * sum(
            (m.q_pv[i] - opfobj.pv_q_init[i]) * cost_pv_q_b[i] * (m.q_pv[i] - opfobj.pv_q_init[i]) for i in pv_node_idx) \
               + m.p_net_import*cost_p_gen_a + 0.5 * m.p_net_import*cost_p_gen_b*m.p_net_import \
               + m.q_net_import*cost_q_gen_a  + 0.5 * m.q_net_import * cost_q_gen_b * m.q_net_import

    if u_unit == 'kW':
        conv_factor = 1000  # to convert from kW to Watt
    elif u_unit == 'MW':
        conv_factor = 1000 * 1000  # to convert from MW to Watt
    else:
        conv_factor = 1
    no_flex_nodes = len(opfobj.flex_info.names)
    no_cap_nodes = len(opfobj.cap_info.names)
    no_xmers_reg = len(opfobj.tap_info.names)
    no_pv_nodes = len(opfobj.pv_info.names)

    tau = 0.00625  # tap steps
    #TODO: use rated values to set limits for the OPF
    p_flex_load_var_limit_min = opfobj.p_flex_load_init * 0
#     p_flex_load_var_limit_min = -opfobj.p_flex_load_init  / 2
    p_flex_load_var_limit_max = opfobj.p_flex_load_init  / 2
    # cap_power_min = 0 * np.ones(shape=no_cap_nodes) * conv_factor
    # cap_power_max = opfobj.cap_bank_init * random.gauss(2, 0.5) / 2
    # cap_power_max = opfobj.cap_load_inc_matrix.T.dot(opfobj.cap_bank_init) * np.ones(shape=no_cap_nodes) * random.gauss(2, 0.5)
    cap_power_min = -opfobj.cap_bank_init * 0
    cap_power_max = opfobj.cap_bank_init / 2


    xmer_min_taps = -16 * np.ones(shape=no_xmers_reg)
    xmer_max_taps = 16 * np.ones(shape=no_xmers_reg)

    # we assume inverters are oversized
    # opfobj.pv_s_init = np.sqrt(1.1 * opfobj.pv_p_init ** 2 + 1.1 * opfobj.pv_q_init ** 2)

    pv_p_max = opfobj.pv_p_init * 0.25  # will only allow to curtail by quarter of the pv generation
    pv_p_min = -opfobj.pv_p_init * 0.0
    q_2 = opfobj.pv_s_init ** 2 - pv_p_max ** 2
    q_max = np.sqrt(q_2.astype(float))
    pv_q_max = q_max*0.25
    pv_q_min = -q_max*0.25

    # extract appropriate line ratings
    # because sometimes network models don't have proper normal amperes and emergency amperes with line defs
    # line_ratings = np.zeros(shape=(len(opfobj.line_ratings)))
    # for i in range(0, len(opfobj.line_ratings)):
    #     if opfobj.line_ratings[i] < opfobj.i_f_abs[i]:
    #         line_ratings[i] = opfobj.i_f_abs[i] * 3  # 2
    #     else:
    #         line_ratings[i] = opfobj.line_ratings[i] * 3  # 2
    #TODO: use rated values to set limits of the OP
    line_ratings = opfobj.i_f_abs*3
    #
    # for i in range(0, len(opfobj.line_ratings)):
    #     if opfobj.line_ratings[i] < opfobj.i_f_abs[i]:
    #         line_ratings[i] = opfobj.i_f_abs[i] * 3  # 2
    #     else:
    #         line_ratings[i] = opfobj.line_ratings[i] * 3  # 2
    voltage_min_lims = opfobj.vL_abs_dss * 0.90
    voltage_max_lims = opfobj.vL_abs_dss * 1.1

    # logger.debug(f"-- preparing central optimization problem --")
    start_central_prep = timeit.default_timer()
    logger.debug(f"Central problem formulation started")

    model = pyo.ConcreteModel()
    opf_nodes_ln = len(opfobj.vL_abs_dss)
    flex_node_idx = range(0, no_flex_nodes)
    node_idx = range(0, opf_nodes_ln)
    line_idx = range(0, len(opfobj.i_f))
    cap_node_idx = range(0, no_cap_nodes)
    pv_node_idx = range(0, no_pv_nodes)
    # cap_unit_power = 1 * conv_factor  # each capacitor unit kVAR power
    num_xmers_idx = range(0, no_xmers_reg)
    # TODO: calculate the fixed parts of the constriants before hand
    # capacitor variables and constraint
    model.cap_active_number = pyo.Var(cap_node_idx)  # , domain=pyo.Integers)
    model.cons_max_cap = pyo.Constraint(cap_node_idx, rule=cap_max_def)
    model.cons_min_cap = pyo.Constraint(cap_node_idx, rule=cap_min_def)
    # xmer variables and constraints
    model.xmer_tap_number = pyo.Var(num_xmers_idx)  # , domain=pyo.Integers)
    model.cons_max_xmer = pyo.Constraint(num_xmers_idx, rule=xmer_max_def)
    model.cons_min_xmer = pyo.Constraint(num_xmers_idx, rule=xmer_min_def)
    # flex load constraints
    model.p_flex_load_var = pyo.Var(flex_node_idx)
    model.cons_p_flex_lim_max = pyo.Constraint(flex_node_idx, rule=pflex_max_def)
    model.cons_p_flex_lim_min = pyo.Constraint(flex_node_idx, rule=pflex_min_def)
    # pv inverter variables and constraints
    model.p_pv = pyo.Var(pv_node_idx)
    model.q_pv = pyo.Var(pv_node_idx)
    model.cons_p_pv_lim_max = pyo.Constraint(pv_node_idx, rule=pv_p_max_def)
    model.cons_p_pv_lim_min = pyo.Constraint(pv_node_idx, rule=pv_p_min_def)
    model.cons_q_pv_lim_max = pyo.Constraint(pv_node_idx, rule=pv_q_max_def)
    model.cons_q_pv_lim_min = pyo.Constraint(pv_node_idx, rule=pv_q_min_def)
    # voltage approximation constraints
    model.voltage_approx = pyo.Var(node_idx)
    model.cons_volt_approx = pyo.Constraint(node_idx, rule=voltage_approx_def)
    model.voltage_lim_max = pyo.Constraint(node_idx, rule=volt_max_def)
    model.voltage_lim_min = pyo.Constraint(node_idx, rule=volt_min_def)
    # line flow approximation constraints
    model.line_flow_approx = pyo.Var(line_idx)
    model.cons_flow_approx = pyo.Constraint(line_idx, rule=line_flow_approx_def)
    model.cons_flow_lim_max = pyo.Constraint(line_idx, rule=line_flow_lim_max_def)
    model.cons_flow_lim_min = pyo.Constraint(line_idx, rule=line_flow_lim_min_def)
    # active and reactive power import variables for global power balance
    model.p_net_import = pyo.Var()
    model.q_net_import = pyo.Var()
    # power balance constraint - loss inclusion
    model.cons_power_balance_P = pyo.Constraint(rule=power_balance_def_P)
    model.cons_power_balance_Q = pyo.Constraint(rule=power_balance_def_Q)
    # objective function
    model.obj = pyo.Objective(rule=obj_rule, sense=pyo.minimize)

    # solver = pyo.SolverFactory('ipopt')
    # solver = pyo.SolverFactory('mindtpy').solve(model, mip_solver='glpk', nlp_solver='ipopt', tee=True)
    # model.Objective.display()
    # model.display()
    # model.pprint()
    # solver.options['tol'] = 1E-3
    # results= solver.solve(model)
    logger.debug(f'opfobj.p_flex_load_init {opfobj.p_flex_load_init}')
    logger.debug(f'p_flex_min {p_flex_load_var_limit_min}')
    logger.debug(f'p_flex_max {p_flex_load_var_limit_max}')
    logger.debug(f'opfobj.reg_taps_init {opfobj.reg_taps_init}')
    logger.debug(f'reg_max {xmer_max_taps}')
    logger.debug(f'reg_min {xmer_min_taps}')

    logger.debug(f'opfobj.cap_bank_init {opfobj.cap_bank_init}')
    logger.debug(f'cap_max {cap_power_max}')
    logger.debug(f'cap_min {cap_power_min}')

    logger.debug(f'opfobj.pv_p_init {opfobj.pv_p_init}')
    logger.debug(f'pv_p_min {pv_p_min}')
    logger.debug(f'pv_p_max {pv_p_max}')

    logger.debug(f'opfobj.pv_q_init {opfobj.pv_q_init}')
    logger.debug(f'pv_q_min {pv_q_min}')
    logger.debug(f'pv_q_max {pv_q_max}')
    logger.debug(f'opfobj.pv_s_init {opfobj.pv_s_init}')

    stop_central_prep = timeit.default_timer()
    central_prep_times = stop_central_prep - start_central_prep
    logger.debug(f"central problem prepared in {central_prep_times} seconds")
    logger.debug(f"--- solving the central problem --")
    start_central_solve = timeit.default_timer()
    results = pyo.SolverFactory('ipopt').solve(model)  # ipopt solver being used
    stop_central_solve = timeit.default_timer()
    central_solve_times = stop_central_solve - start_central_solve
    logger.debug(f"central problem solved in {stop_central_solve - start_central_solve} seconds")
    central_times = central_solve_times + central_prep_times
    # results = pyo.SolverFactory('xpress_direct').solve(model)
    # results = pyo.SolverFactory('mindtpy').solve(model, mip_solver='glpk', nlp_solver='ipopt', tee=True)

    # central_solution_value[loop] = pyo.value(model.obj)
    logger.debug(f"{results.solver.Message}")
    try:
        opfobj.p_flex_load_var_opti_abs = []
        for i in flex_node_idx:
            opfobj.p_flex_load_var_opti_abs.append(pyo.value(model.p_flex_load_var[i]))
        volt_approx_opti = []
        for i in node_idx:
            volt_approx_opti.append(pyo.value(model.voltage_approx[i]))
        flow_approx_opti = []
        for l in line_idx:
            flow_approx_opti.append(pyo.value(model.line_flow_approx[l]))
        opfobj.cap_value_opti_abs = []
        for c in cap_node_idx:
            opfobj.cap_value_opti_abs.append(pyo.value(model.cap_active_number[c]))
        opfobj.xmer_value_opti_abs = []
        for x in num_xmers_idx:
            opfobj.xmer_value_opti_abs.append(pyo.value(model.xmer_tap_number[x]))
        opfobj.p_pv_opti_abs = []
        for i in pv_node_idx:
            opfobj.p_pv_opti_abs.append(pyo.value(model.p_pv[i]))
        opfobj.q_pv_opti_abs = []
        for i in pv_node_idx:
            opfobj.q_pv_opti_abs.append(pyo.value(model.q_pv[i]))

        #TODO: check validation script below

        opfobj.active_power_generation_import = pyo.value(model.p_net_import)
        opfobj.reactive_power_generation_import = pyo.value(model.q_net_import)
        # array_p_flex_load_var_opti = np.array(opfobj.p_flex_load_var_opti_abs).transpose()
        # array_p_pv_opti = np.array(opfobj.p_pv_opti_abs).transpose()
        # array_q_pv_opti = np.array(opfobj.q_pv_opti_abs).transpose()
        # array_q_cap_opti = np.array(opfobj.cap_value_opti_abs)
        # array_tap_opti = np.array(opfobj.xmer_value_opti_abs)
        # # some validation scripts
        # # active power
        # cap_p_losses = opfobj.m_pl_qY.dot(opfobj.cap_load_inc_matrix).dot(np.array(opfobj.cap_value_opti))
        # flex_load_p_losses = opfobj.m_pl_pY.dot(opfobj.flex_load_inc_matrix).dot(np.array(opfobj.p_flex_load_var_opti))
        # pv_p_plosses = opfobj.m_pl_pY.dot(opfobj.pv_inc_matrix).dot(np.array(opfobj.p_pv_opti))
        # pv_q_plosses = opfobj.m_pl_qY.dot(opfobj.pv_inc_matrix).dot(np.array(opfobj.q_pv_opti))
        # total_p_load = sum(opfobj.pL) - sum(np.array(opfobj.p_flex_load_var_opti)) - sum(np.array(opfobj.p_pv_opti))
        # total_p_losses = pv_q_plosses + pv_p_plosses + cap_p_losses + flex_load_p_losses + opfobj.p_loss_dss
        # net_gen_p_imports = (total_p_load + total_p_losses) / 1000
        # net_p_original_imports = (sum(opfobj.pL) + opfobj.p_loss_dss) / 1000
        # # reactive power
        # cap_q_losses = opfobj.m_ql_qY.dot(opfobj.cap_load_inc_matrix).dot(np.array(opfobj.cap_value_opti))
        # flex_load_q_losses = opfobj.m_ql_pY.dot(opfobj.flex_load_inc_matrix).dot(np.array(opfobj.p_flex_load_var_opti))
        # pv_p_qlosses = opfobj.m_ql_pY.dot(opfobj.pv_inc_matrix).dot(np.array(opfobj.p_pv_opti))
        # pv_q_qlosses = opfobj.m_ql_qY.dot(opfobj.pv_inc_matrix).dot(np.array(opfobj.q_pv_opti))
        # total_q_load = sum(opfobj.qL) - sum(np.array(opfobj.cap_value_opti)) - sum(np.array(opfobj.q_pv_opti))
        # total_q_losses = pv_p_qlosses + pv_q_qlosses + cap_q_losses + flex_load_q_losses + opfobj.q_loss_dss
        # net_gen_q_imports = (total_q_load + total_q_losses) / 1000
        # net_q_original_imports = (sum(opfobj.qL) + opfobj.q_loss_dss) / 1000

        logger.debug(f'Optimization Results Loaded')

    except:
        logger.debug(f'Optimization Results not Loaded there may be some errors')
        logger.debug(f"using base-values")
        opfobj.active_power_generation_import = np.genfromtxt('base_vals/active_power_generation_import.csv', delimiter=',')
        opfobj.reactive_power_generation_import = np.genfromtxt('base_vals/reactive_power_generation_import.csv', delimiter=',')
        opfobj.p_flex_load_var_opti_abs = np.genfromtxt('base_vals/array_p_flex_load_var_opti.csv', delimiter=',')
        opfobj.p_pv_opti_abs = np.genfromtxt('base_vals/array_p_pv_opti.csv', delimiter=',')
        opfobj.q_pv_opti_abs = np.genfromtxt('base_vals/array_q_pv_opti.csv', delimiter=',')
        opfobj.cap_value_opti_abs = np.genfromtxt('base_vals/array_q_cap_opti.csv', delimiter=',')
        opfobj.xmer_value_opti_abs = np.genfromtxt('base_vals/xmer_value_opti.csv', delimiter=',')



    logger.debug("======= initial values loaded in the optimization ======== ")
    logger.debug(f'opfobj.p_flex_load_init {opfobj.p_flex_load_init}')
    logger.debug(f'opfobj.reg_taps_init {opfobj.reg_taps_init}')
    logger.debug(f'opfobj.cap_bank_init {opfobj.cap_bank_init}')
    logger.debug(f'opfobj.pv_p_init {opfobj.pv_p_init}')
    logger.debug(f'opfobj.pv_q_init {opfobj.pv_q_init}')
    logger.debug(f'opfobj.pv_s_init {opfobj.pv_s_init}')


    logger.debug("======= optimal variable movement proposed ==== ")
    logger.debug(f'active_power_generation_import {opfobj.active_power_generation_import}')
    logger.debug(f'reactive_power_generation_import {opfobj.reactive_power_generation_import}')
    logger.debug(f'array_p_flex_load_var_opti_abs {opfobj.p_flex_load_var_opti_abs}')
    logger.debug(f'array_p_pv_opti_abs { opfobj.p_pv_opti_abs}')
    logger.debug(f'array_q_pv_opti_abs {opfobj.q_pv_opti_abs}')
    logger.debug(f'array_q_cap_opti_abs {opfobj.cap_value_opti_abs}')
    logger.debug(f'array_tap_opti_abs {opfobj.xmer_value_opti_abs}')

    logger.debug(f"===== sainty checks =====")
    for j in range(0, len(opfobj.p_flex_load_var_opti_abs)):
        if opfobj.p_flex_load_var_opti_abs[j] < p_flex_load_var_limit_min[j]:
            opfobj.p_flex_load_var_opti_abs[j] = p_flex_load_var_limit_min[j]
        elif opfobj.p_flex_load_var_opti_abs[j] > p_flex_load_var_limit_max[j]:
            opfobj.p_flex_load_var_opti_abs[j] = p_flex_load_var_limit_max[j]

    for j in range(0, len(opfobj.reg_taps_init)):
        if opfobj.xmer_value_opti_abs[j] < xmer_min_taps[j]:
            opfobj.xmer_value_opti_abs[j] = xmer_min_taps[j]
        elif opfobj.xmer_value_opti_abs[j] > xmer_max_taps[j]:
            opfobj.xmer_value_opti_abs[j] = xmer_max_taps[j]

    for j in range(0, len(opfobj.cap_bank_init)):
        if opfobj.cap_value_opti_abs[j] < cap_power_min[j]:
            opfobj.cap_value_opti_abs[j] = cap_power_min[j]
        elif opfobj.cap_value_opti_abs[j] > cap_power_max[j]:
            opfobj.cap_value_opti_abs[j] = cap_power_max[j]

    for j in range(0, len(opfobj.pv_q_init)):
        if opfobj.q_pv_opti_abs[j] < pv_q_min[j]:
            opfobj.q_pv_opti_abs[j] = pv_q_min[j]
        elif opfobj.q_pv_opti_abs[j] > pv_q_max[j]:
            opfobj.q_pv_opti_abs[j] = pv_q_max[j]

    for j in range(0, len(opfobj.pv_p_init)):
        if opfobj.p_pv_opti_abs[j] < pv_p_min[j]:
            opfobj.p_pv_opti_abs[j] = pv_p_min[j]
        elif opfobj.p_pv_opti_abs[j] > pv_p_max[j]:
            opfobj.p_pv_opti_abs[j] = pv_p_max[j]

    logger.debug("======= optimal variable movement proposed after sanity checks==== ")
    logger.debug(f'active_power_generation_import {opfobj.active_power_generation_import}')
    logger.debug(f'reactive_power_generation_import {opfobj.reactive_power_generation_import}')
    logger.debug(f'array_p_flex_load_var_opti_abs {opfobj.p_flex_load_var_opti_abs}')
    logger.debug(f'array_p_pv_opti_abs { opfobj.p_pv_opti_abs}')
    logger.debug(f'array_q_pv_opti_abs {opfobj.q_pv_opti_abs}')
    logger.debug(f'array_q_cap_opti_abs {opfobj.cap_value_opti_abs}')
    logger.debug(f'array_tap_opti_abs {opfobj.xmer_value_opti_abs}')


    opfobj.p_flex_load_var_opti = opfobj.p_flex_load_init - np.array(opfobj.p_flex_load_var_opti_abs)
    opfobj.cap_value_opti = opfobj.cap_bank_init - np.array(opfobj.cap_value_opti_abs)
    opfobj.xmer_value_opti = opfobj.reg_taps_init - np.array(opfobj.xmer_value_opti_abs)
    opfobj.p_pv_opti = opfobj.pv_p_init - np.array(opfobj.p_pv_opti_abs)
    opfobj.q_pv_opti =  opfobj.pv_q_init - np.array(opfobj.q_pv_opti_abs)

    logger.debug("======= control variable movement sent to feeder (Current Val - Optimal Val) ==== ")

    logger.debug(f'array_p_flex_load_var_opti_to_send {opfobj.p_flex_load_var_opti}')
    logger.debug(f'cap_value_opti_to_send {opfobj.cap_value_opti}')
    logger.debug(f'xmer_value_opti_to_send {opfobj.xmer_value_opti}')
    logger.debug(f'p_pv_opti_to_send {opfobj.p_pv_opti}')
    logger.debug(f'q_pv_opti_to_send {opfobj.q_pv_opti}')



def distributed_optimization(opfobj):
    alpha_p = 1e-3
    alpha_q = 1e-3
    k = 0
    count = 5000
    tolerance = 1e-1
    term = 0
    # Tuning variables
    alpha_mu_v_max = 1e-3
    alpha_mu_v_min = 1e-3
    alpha_mu_i_ij_max = 1e-3
    alpha_mu_i_ij_min = 1e-3
    alpha_mu_p_flex_max = 1e-3
    alpha_mu_p_flex_min = 1e-3
    alpha_mu_p_pv_max = 1e-3
    alpha_mu_p_pv_min = 1e-3
    alpha_mu_q_pv_max = 1e-3
    alpha_mu_q_pv_min = 1e-3
    alpha_mu_cap_max = 1e-3
    alpha_mu_cap_min = 1e-3
    alpha_mu_xmer_max = 1e-3
    alpha_mu_xmer_min = 1e-3
    # some initializing stuff
    p_flex_load_desired = opfobj.p_flex_load_init / 2
    reg_taps_desired = opfobj.reg_taps_init / 2
    cap_bank_desired = opfobj.cap_load_inc_matrix.T.dot(opfobj.cap_bank_init)
    pv_p_desired = opfobj.pv_inc_matrix.T.dot(opfobj.pv_p_init)
    pv_q_desired = opfobj.pv_inc_matrix.T.dot(opfobj.pv_q_init)
    max_line_ratings = line_ratings
    min_line_ratings = -line_ratings
    # Primal variables
    # Flex load Set points
    p_flex_load_dist = np.zeros(shape=(no_flex_nodes, 5000 + 1))
    p_flex_load_dist[:, 0] = p_flex_load_desired  # initialized to the desired/rated state
    # cap set points
    cap_active_dist = np.zeros(shape=(no_cap_nodes, 5000 + 1))
    cap_active_dist[:, 0] = cap_bank_desired  # initialized to desired/rated state
    # xmer set points
    xmer_tap_dist = np.zeros(shape=(no_xmers_reg, 5000 + 1))
    xmer_tap_dist[:, 0] = reg_taps_desired  # initialized to the desired/rated state
    # pv active power setpoints
    p_pv_dist = np.zeros(shape=(no_pv_nodes, 5000 + 1))
    p_pv_dist[:, 0] = pv_p_desired  # initialized to the desired/rated state
    # pv reactive power setpoint
    q_pv_dist = np.zeros(shape=(no_pv_nodes, 5000 + 1))
    q_pv_dist[:, 0] = pv_q_desired  # initialized to the desired/rated state
    # Dual variables
    # power balance dual
    lambda_p = np.zeros(shape=(1, 5000 + 1))
    lambda_q = np.zeros(shape=(1, 5000 + 1))
    p_net_import_dist = np.zeros(shape=(1, 5000 + 1))
    p_net_import_dist[:, 0] = sum(opfobj.pL) + opfobj.p_loss_dss - sum(
        pv_p_desired)  # initialized to the rated state
    q_net_import_dist = np.zeros(shape=(1, 5000 + 1))
    p_net_import_dist[:, 0] = sum(opfobj.qL) + opfobj.q_loss_dss - sum(
        pv_q_desired)  # initialized to the rated state

    # Bus voltage magnitude dual
    mu_v_max = np.zeros(shape=(len(node_idx), 5000 + 1))
    mu_v_min = np.zeros(shape=(len(node_idx), 5000 + 1))

    # Branch Flows duals
    mu_i_ij_max = np.zeros(shape=(len(line_idx), 5000 + 1))
    mu_i_ij_min = np.zeros(shape=(len(line_idx), 5000 + 1))

    # Generation duals
    mu_p_flex_max = np.zeros(shape=(no_flex_nodes, 5000 + 1))
    mu_p_flex_min = np.zeros(shape=(no_flex_nodes, 5000 + 1))

    # cap limit duals
    mu_cap_max = np.zeros(shape=(no_cap_nodes, 5000 + 1))
    mu_cap_min = np.zeros(shape=(no_cap_nodes, 5000 + 1))

    # xmer limit duals
    mu_xmer_max = np.zeros(shape=(no_xmers_reg, 5000 + 1))
    mu_xmer_min = np.zeros(shape=(no_xmers_reg, 5000 + 1))

    # PV Active power duals
    mu_p_pv_max = np.zeros(shape=(no_pv_nodes, 5000 + 1))
    mu_p_pv_min = np.zeros(shape=(no_pv_nodes, 5000 + 1))

    # PV Reactove power duals
    mu_q_pv_max = np.zeros(shape=(no_pv_nodes, 5000 + 1))
    mu_q_pv_min = np.zeros(shape=(no_pv_nodes, 5000 + 1))

    residual_v_max_viol_idx = np.empty(shape=(5000 + 1), dtype='bool')
    residual_v_min_viol_idx = np.empty(shape=(5000 + 1), dtype='bool')
    residual_ij_max_viol_idx = np.empty(shape=(5000 + 1), dtype='bool')
    residual_ij_min_viol_idx = np.empty(shape=(5000 + 1), dtype='bool')
    residual_cap_max_viol_idx = np.empty(shape=(5000 + 1), dtype='bool')
    residual_cap_min_viol_idx = np.empty(shape=(5000 + 1), dtype='bool')
    residual_xmer_max_viol_idx = np.empty(shape=(5000 + 1), dtype='bool')
    residual_xmer_min_viol_idx = np.empty(shape=(5000 + 1), dtype='bool')
    residual_bal_P = np.zeros(shape=(5000 + 1))
    residual_bal_Q = np.zeros(shape=(5000 + 1))

    start_distributed = timeit.default_timer()

    while (k < count) and (term == 0):
        """
            equality active and reactive power constraints are of the form 
            def power_balance_def_P(m):
                p_loss_p_flex_matrix = opfobj.m_pl_pY.dot(opfobj.flex_load_inc_matrix)
                p_loss_cap_matrix = opfobj.m_pl_qY.dot(opfobj.cap_load_inc_matrix)
                p_loss_flex_expr = (p_loss_p_flex_matrix[j] * (m.p_flex_load_var[j]) for j in flex_node_idx)
                p_loss_cap_expr = (p_loss_cap_matrix[j] * (m.cap_active_number[j]) for j in cap_node_idx)
                p_fixed_load_sum = sum(opfobj.pL[j] for j in node_idx)
                p_flex_load_sum = sum(m.p_flex_load_var[j] for j in flex_node_idx)
                return m.p_net_import \
                       == p_fixed_load_sum \
                       - p_flex_load_sum \
                       + sum(p_loss_cap_expr) \
                       + sum(p_loss_flex_expr) \
                       + opfobj.p_loss_dss
        """
        # First - order Derivatives
        dL_dLambda_P = \
            -p_net_import_dist[:, k] \
            + sum(opfobj.pL) \
            - sum(p_flex_load_dist[:, k]) \
            - sum(p_pv_dist[:, k]) \
            + opfobj.m_pl_pY.dot(opfobj.flex_load_inc_matrix.dot(p_flex_load_dist[:, k])) \
            + opfobj.m_pl_qY.dot(opfobj.cap_load_inc_matrix.dot(cap_active_dist[:, k])) \
            + opfobj.m_pl_pY.dot(opfobj.pv_inc_matrix.dot(p_pv_dist[:, k])) \
            + opfobj.m_pl_qY.dot(opfobj.pv_inc_matrix.dot(q_pv_dist[:, k])) \
            + opfobj.p_loss_dss

        """
        Equality reactive power constraints are of the form
        def power_balance_def_Q(m):
            q_loss_p_flex_matrix = opfobj.m_ql_pY.dot(opfobj.flex_load_inc_matrix)
            q_loss_cap_matrix = opfobj.m_ql_qY.dot(opfobj.cap_load_inc_matrix)
            q_loss_flex_expr = (q_loss_p_flex_matrix[j] * (m.p_flex_load_var[j]) for j in flex_node_idx)
            q_loss_cap_expr = (q_loss_cap_matrix[j] * (m.cap_active_number[j]) for j in cap_node_idx)
            q_cap_flex_expr = sum(m.cap_active_number[j] for j in cap_node_idx)
            q_fixed_load_expr = sum(opfobj.qL[j] for j in node_idx)
            return m.q_net_import \
                   == q_fixed_load_expr \
                   - q_cap_flex_expr \
                   + sum(q_loss_cap_expr) \
                   + sum(q_loss_flex_expr) \
                   + opfobj.q_loss_dss
        """
        dL_dLambda_Q = -q_net_import_dist[:, k] \
                       + sum(opfobj.qL) \
                       - sum(cap_active_dist[:, k]) \
                       - sum(q_pv_dist[:, k]) \
                       + opfobj.m_ql_pY.dot(opfobj.flex_load_inc_matrix.dot(p_flex_load_dist[:, k])) \
                       + opfobj.m_ql_qY.dot(opfobj.cap_load_inc_matrix.dot(cap_active_dist[:, k])) \
                       + opfobj.m_ql_pY.dot(opfobj.pv_inc_matrix.dot(p_pv_dist[:, k])) \
                       + opfobj.m_ql_qY.dot(opfobj.pv_inc_matrix.dot(q_pv_dist[:, k])) \
                       + opfobj.q_loss_dss

        dL_dmu_ij_max = max_line_ratings - (opfobj.i_f_abs
                                            + opfobj.M_if_pY.dot(
                    opfobj.flex_load_inc_matrix.dot(p_flex_load_dist[:, k]))
                                            + opfobj.M_if_qY.dot(
                    opfobj.cap_load_inc_matrix.dot(cap_active_dist[:, k])))

        dL_dmu_ij_min = (opfobj.i_f_abs
                         + opfobj.M_if_qY.dot(opfobj.cap_load_inc_matrix.dot(cap_active_dist[:, k]))
                         + opfobj.M_if_pY.dot(
                    opfobj.flex_load_inc_matrix.dot(p_flex_load_dist[:, k]))) - min_line_ratings

        dL_dmu_v_max = voltage_max_lims - (opfobj.vL_abs_dss
                                           + np.diag(opfobj.vL_abs_dss).dot(
                    tau * opfobj.reg_inc_matrix.dot(opfobj.reg_taps_init - xmer_tap_dist[:, k]))
                                           + opfobj.K_vMag_qY.dot(
                    opfobj.cap_load_inc_matrix.dot(cap_active_dist[:, k]))
                                           + opfobj.K_vMag_pY.dot(
                    opfobj.flex_load_inc_matrix.dot(p_flex_load_dist[:, k])))

        dL_dmu_v_min = (opfobj.vL_abs_dss
                        + np.diag(opfobj.vL_abs_dss).dot(
                    tau * opfobj.reg_inc_matrix.dot(opfobj.reg_taps_init - xmer_tap_dist[:, k]))
                        + opfobj.K_vMag_qY.dot(opfobj.cap_load_inc_matrix.dot(cap_active_dist[:, k]))
                        + opfobj.K_vMag_pY.dot(
                    opfobj.flex_load_inc_matrix.dot(p_flex_load_dist[:, k]))) - voltage_min_lims

        # dL_dmu_pg_max = -p_flex_load_dist[:, k] + p_flex_load_var_limit_max
        # dL_dmu_pg_min = p_flex_load_dist[:, k] - p_flex_load_var_limit_min
        # dL_dmu_cap_max = -cap_active_dist[:, k] + cap_power_max
        # dL_dmu_cap_min = cap_active_dist[:, k] - cap_power_min
        dL_dmu_xmer_max = -xmer_tap_dist[:, k] + xmer_max_taps
        dL_dmu_xmer_min = xmer_tap_dist[:, k] - xmer_min_taps
        # Primal Updates
        # flex_load update
        temp_p_flex = -np.array(cost_a).reshape((len(flex_node_idx))) + lambda_p[:, k] * opfobj.m_pl_pY.dot(
            opfobj.flex_load_inc_matrix) \
                      - lambda_p[:, k] * np.ones(shape=(len(flex_node_idx))) + lambda_q[:, k] * opfobj.m_ql_pY.dot(
            opfobj.flex_load_inc_matrix) \
                      - opfobj.flex_load_inc_matrix.transpose().dot(opfobj.M_if_pY).dot(
            mu_i_ij_max[:, k] - mu_i_ij_min[:, k]) \
                      - opfobj.flex_load_inc_matrix.transpose().dot(opfobj.K_vMag_pY).dot(
            mu_v_max[:, k] - mu_v_min[:, k])
        # + (mu_p_flex_max[:, k] - mu_p_flex_min[:, k])

        p_flex_load_dist_temp = p_flex_load_desired + np.diag(np.array(1 / (cost_b))).dot(temp_p_flex)

        # capacitor banks
        temp_q_cap = -lambda_q[:, k] * np.ones(shape=(len(cap_node_idx))) + \
                     + lambda_p[:, k] * opfobj.m_pl_qY.dot(opfobj.cap_load_inc_matrix) \
                     + lambda_q[:, k] * opfobj.m_ql_qY.dot(opfobj.cap_load_inc_matrix) \
                     - opfobj.cap_load_inc_matrix.transpose().dot(opfobj.K_vMag_qY).dot(
            mu_v_max[:, k] - mu_v_min[:, k]) \
                     - opfobj.cap_load_inc_matrix.transpose().dot(opfobj.M_if_qY).dot(
            mu_i_ij_max[:, k] - mu_i_ij_min[:, k])
        # + (mu_cap_max[:, k] - mu_cap_min[:, k])

        cap_active_dist_temp = cap_bank_desired + np.diag(np.array(1 / (cost_bank_b))).dot(temp_q_cap)

        # transformer regulators
        temp_xmer = (np.diag(opfobj.vL_abs_dss).dot(tau * opfobj.reg_inc_matrix)).transpose().dot(
            mu_v_max[:, k] - mu_v_min[:, k])  # \
        # + (mu_xmer_max[:, k] - mu_xmer_min[:, k])
        xmer_active_dist_temp = reg_taps_desired + np.diag(np.array(1 / (cost_reg_b))).dot(temp_xmer)

        # flex_load active power update
        temp_p_pv = -np.array(cost_pv_p_a).reshape((len(pv_node_idx))) + lambda_p[:, k] * opfobj.m_pl_pY.dot(
            opfobj.pv_inc_matrix) \
                    - lambda_p[:, k] * np.ones(shape=(len(pv_node_idx))) + lambda_q[:, k] * opfobj.m_ql_pY.dot(
            opfobj.pv_inc_matrix) \
                    - opfobj.pv_inc_matrix.transpose().dot(opfobj.M_if_pY).dot(
            mu_i_ij_max[:, k] - mu_i_ij_min[:, k]) \
                    - opfobj.pv_inc_matrix.transpose().dot(opfobj.K_vMag_pY).dot(mu_v_max[:, k] - mu_v_min[:, k])

        p_pv_dist_temp = pv_p_desired + np.diag(np.array(1 / (cost_pv_p_b))).dot(temp_p_pv)

        # flex_load reactive power updates
        temp_q_pv = -lambda_q[:, k] * np.ones(shape=(len(pv_node_idx))) + \
                    + lambda_p[:, k] * opfobj.m_pl_qY.dot(opfobj.pv_inc_matrix) \
                    + lambda_q[:, k] * opfobj.m_ql_qY.dot(opfobj.pv_inc_matrix) \
                    - opfobj.pv_inc_matrix.transpose().dot(opfobj.K_vMag_qY).dot(mu_v_max[:, k] - mu_v_min[:, k]) \
                    - opfobj.pv_inc_matrix.transpose().dot(opfobj.M_if_qY).dot(
            mu_i_ij_max[:, k] - mu_i_ij_min[:, k])

        q_pv_dist_temp = pv_q_desired + np.diag(np.array(1 / (cost_pv_q_b))).dot(temp_q_pv)

        # generation imports
        p_net_import_dist[:, k + 1] = (-lambda_p[:, k] - cost_p_gen_a) / cost_p_gen_b
        q_net_import_dist[:, k + 1] = (-lambda_q[:, k] - cost_q_gen_a) / cost_q_gen_b
        # Projecting flexible loads within their
        # limits using a stationary point of its optimality condition
        max_viol_mask = [p_flex_load_dist_temp[i] > p_flex_load_var_limit_max[i] for i in
                         range(len(p_flex_load_var_limit_max))]
        min_viol_mask = [p_flex_load_dist_temp[i] < p_flex_load_var_limit_min[i] for i in
                         range(len(p_flex_load_var_limit_min))]
        p_flex_load_dist_temp[max_viol_mask] = p_flex_load_var_limit_max[max_viol_mask]
        p_flex_load_dist_temp[min_viol_mask] = p_flex_load_var_limit_min[min_viol_mask]
        p_flex_load_dist[:, k + 1] = p_flex_load_dist_temp

        # projecting PV Active Power
        max_viol_mask = [p_pv_dist_temp[i] > pv_p_max[i] for i in
                         range(len(pv_p_max))]
        min_viol_mask = [p_pv_dist_temp[i] < pv_p_min[i] for i in
                         range(len(pv_p_min))]
        p_pv_dist_temp[max_viol_mask] = pv_p_max[max_viol_mask]
        p_pv_dist_temp[min_viol_mask] = pv_p_min[min_viol_mask]
        p_pv_dist[:, k + 1] = p_pv_dist_temp

        # projecting PV Reactive Power
        max_viol_mask = [q_pv_dist_temp[i] > pv_q_max[i] for i in
                         range(len(pv_q_max))]
        min_viol_mask = [q_pv_dist_temp[i] < pv_q_min[i] for i in
                         range(len(pv_q_min))]
        q_pv_dist_temp[max_viol_mask] = pv_q_max[max_viol_mask]
        q_pv_dist_temp[min_viol_mask] = pv_q_min[min_viol_mask]
        q_pv_dist[:, k + 1] = q_pv_dist_temp

        # limits for generation and load import
        # if p_net_import_dist[:, k+1] < 0:
        #     p_net_import_dist[:, k + 1] = 0
        # if q_net_import_dist[:, k+1] < 0:
        #     q_net_import_dist[:, k + 1] = 0

        # projecting cap values
        max_viol_cap = [cap_active_dist_temp[i] > (cap_power_max[i]) for i in range(len(cap_power_max))]
        min_viol_cap = [cap_active_dist_temp[i] < (cap_power_min[i]) for i in range(len(cap_power_min))]
        cap_active_dist_temp[max_viol_cap] = cap_power_max[max_viol_cap]
        cap_active_dist_temp[min_viol_cap] = cap_power_min[min_viol_cap]
        cap_active_dist[:, k + 1] = cap_active_dist_temp
        # projecting xmer values
        max_viol_xmer = [xmer_active_dist_temp[i] > xmer_max_taps[i] for i in range(len(xmer_max_taps))]
        min_viol_xmer = [xmer_active_dist_temp[i] < xmer_min_taps[i] for i in range(len(xmer_min_taps))]
        xmer_active_dist_temp[max_viol_xmer] = xmer_max_taps[max_viol_xmer]
        xmer_active_dist_temp[min_viol_xmer] = xmer_min_taps[min_viol_xmer]
        xmer_tap_dist[:, k + 1] = xmer_active_dist_temp
        # Dual updates
        # lambda update
        lambda_p[:, k + 1] = lambda_p[:, k] - alpha_p * dL_dLambda_P
        lambda_q[:, k + 1] = lambda_q[:, k] - alpha_q * dL_dLambda_Q
        # line limit dual variables updates and projections
        mu_ij_max_temp = mu_i_ij_max[:, k] - alpha_mu_i_ij_max * dL_dmu_ij_max
        #    projecting if max line variable dual has a neg value
        mu_ij_max_temp[mu_ij_max_temp < 0] = 0
        mu_i_ij_max[:, k + 1] = mu_ij_max_temp

        mu_ij_min_temp = mu_i_ij_min[:, k] - alpha_mu_i_ij_min * dL_dmu_ij_min
        #    projecting if max line variable dual has a neg value
        mu_ij_min_temp[mu_ij_min_temp < 0] = 0
        mu_i_ij_min[:, k + 1] = mu_ij_min_temp

        # voltage limit dual variables updates and projections
        mu_v_max_temp = mu_v_max[:, k] - alpha_mu_v_max * dL_dmu_v_max
        #    projecting if max voltage variable dual has a neg value
        mu_v_max_temp[mu_v_max_temp < 0] = 0
        mu_v_max[:, k + 1] = mu_v_max_temp

        mu_v_min_temp = mu_v_min[:, k] - alpha_mu_v_min * dL_dmu_v_min
        #    projecting if min voltage variable dual has a neg value
        mu_v_min_temp[mu_v_min_temp < 0] = 0
        mu_v_min[:, k + 1] = mu_v_min_temp

        # # generation limit dual variables updates and projections
        # mu_p_flex_max_temp = mu_p_flex_max[:, k] - alpha_mu_p_flex_max * dL_dmu_pg_max
        # #    projecting if max flex load variable dual has a neg value
        # mu_p_flex_max_temp[mu_p_flex_max_temp < 0] = 0
        # mu_p_flex_max[:, k + 1] = mu_p_flex_max_temp
        #
        # mu_p_flex_min_temp = mu_p_flex_min[:, k] - alpha_mu_p_flex_min * dL_dmu_pg_min
        # #    projecting if min flex load variable dual has a neg value
        # mu_p_flex_min_temp[mu_p_flex_min_temp < 0] = 0
        # mu_p_flex_min[:, k + 1] = mu_p_flex_min_temp
        #
        # # cap limit dual variables updates and projections
        # mu_cap_max_temp = mu_cap_max[:, k] - alpha_mu_cap_max * dL_dmu_cap_max
        # #    projecting if max cap variable dual has a neg value
        # mu_cap_max_temp[mu_cap_max_temp < 0] = 0
        # mu_cap_max[:, k + 1] = mu_cap_max_temp
        #
        # mu_cap_min_temp = mu_cap_min[:, k] - alpha_mu_cap_min * dL_dmu_cap_min
        # #    projecting if min cap variable dual has a neg value
        # mu_cap_min_temp[mu_cap_min_temp < 0] = 0
        # mu_cap_min[:, k + 1] = mu_cap_min_temp
        #
        # xmer limit dual variables updates and projections
        mu_xmer_max_temp = mu_xmer_max[:, k] - alpha_mu_xmer_max * dL_dmu_xmer_max
        #    projecting if max xmer variable dual has a neg value
        mu_xmer_max_temp[mu_xmer_max_temp < 0] = 0
        mu_xmer_max[:, k + 1] = mu_xmer_max_temp

        mu_xmer_min_temp = mu_xmer_min[:, k] - alpha_mu_xmer_min * dL_dmu_xmer_min
        #    projecting if min cap variable dual has a neg value
        mu_xmer_min_temp[mu_xmer_min_temp < 0] = 0
        mu_xmer_min[:, k + 1] = mu_xmer_min_temp
        """
            equality active and reactive power constraints are of the form 
            def power_balance_def_P(m):
                p_loss_p_flex_matrix = opfobj.m_pl_pY.dot(opfobj.flex_load_inc_matrix)
                p_loss_cap_matrix = opfobj.m_pl_qY.dot(opfobj.cap_load_inc_matrix)
                p_loss_flex_expr = (p_loss_p_flex_matrix[j] * (m.p_flex_load_var[j]) for j in flex_node_idx)
                p_loss_cap_expr = (p_loss_cap_matrix[j] * (m.cap_active_number[j]) for j in cap_node_idx)
                p_fixed_load_sum = sum(opfobj.pL[j] for j in node_idx)
                p_flex_load_sum = sum(m.p_flex_load_var[j] for j in flex_node_idx)
                return m.p_net_import \
                       == p_fixed_load_sum \
                       - p_flex_load_sum \
                       + sum(p_loss_cap_expr) \
                       + sum(p_loss_flex_expr) \
                       + opfobj.p_loss_dss
        """
        residual_bal_P[k + 1] = -p_net_import_dist[:, k] \
                                + sum(opfobj.pL) \
                                - sum(p_flex_load_dist[:, k]) \
                                - sum(p_pv_dist[:, k]) \
                                + opfobj.m_pl_pY.dot(opfobj.flex_load_inc_matrix.dot(p_flex_load_dist[:, k])) \
                                + opfobj.m_pl_qY.dot(opfobj.cap_load_inc_matrix.dot(cap_active_dist[:, k])) \
                                + opfobj.m_pl_pY.dot(opfobj.pv_inc_matrix.dot(p_pv_dist[:, k])) \
                                + opfobj.m_pl_qY.dot(opfobj.pv_inc_matrix.dot(q_pv_dist[:, k])) \
                                + opfobj.p_loss_dss
        """

         Equality reactive power constraints are of the form
         def power_balance_def_Q(m):
             q_loss_p_flex_matrix = opfobj.m_ql_pY.dot(opfobj.flex_load_inc_matrix)
             q_loss_cap_matrix = opfobj.m_ql_qY.dot(opfobj.cap_load_inc_matrix)
             q_loss_flex_expr = (q_loss_p_flex_matrix[j] * (m.p_flex_load_var[j]) for j in flex_node_idx)
             q_loss_cap_expr = (q_loss_cap_matrix[j] * (m.cap_active_number[j]) for j in cap_node_idx)
             q_cap_flex_expr = sum(m.cap_active_number[j] for j in cap_node_idx)
             q_fixed_load_expr = sum(opfobj.qL[j] for j in node_idx)
             return m.q_net_import \
                    == q_fixed_load_expr \
                    - q_cap_flex_expr \
                    + sum(q_loss_cap_expr) \
                    + sum(q_loss_flex_expr) \
                    + opfobj.q_loss_dss

         """

        residual_bal_Q[k + 1] = -q_net_import_dist[:, k] \
                                + sum(opfobj.qL) \
                                - sum(cap_active_dist[:, k]) \
                                - sum(q_pv_dist[:, k]) \
                                + opfobj.m_ql_pY.dot(opfobj.flex_load_inc_matrix.dot(p_flex_load_dist[:, k])) \
                                + opfobj.m_ql_qY.dot(opfobj.cap_load_inc_matrix.dot(cap_active_dist[:, k])) \
                                + opfobj.m_ql_pY.dot(opfobj.pv_inc_matrix.dot(p_pv_dist[:, k])) \
                                + opfobj.m_ql_qY.dot(opfobj.pv_inc_matrix.dot(q_pv_dist[:, k])) \
                                + opfobj.q_loss_dss

        voltage_expr = opfobj.vL_abs_dss \
                       + opfobj.K_vMag_pY.dot(opfobj.flex_load_inc_matrix.dot(p_flex_load_dist[:, k])) \
                       + opfobj.K_vMag_pY.dot(opfobj.pv_inc_matrix.dot(p_pv_dist[:, k])) \
                       + np.diag(opfobj.vL_abs_dss).dot(
            tau * opfobj.reg_inc_matrix.dot(opfobj.reg_taps_init - xmer_tap_dist[:, k])) \
                       + opfobj.K_vMag_qY.dot(opfobj.cap_load_inc_matrix.dot(cap_active_dist[:, k])) \
                       + opfobj.K_vMag_qY.dot(opfobj.pv_inc_matrix.dot(q_pv_dist[:, k]))

        current_expr = opfobj.i_f_abs \
                       + opfobj.M_if_pY.dot(- opfobj.flex_load_inc_matrix.dot(p_flex_load_dist[:, k])) \
                       + opfobj.M_if_qY.dot(opfobj.cap_load_inc_matrix.dot(cap_active_dist[:, k])) \
                       + opfobj.M_if_pY.dot(- opfobj.pv_inc_matrix.dot(p_pv_dist[:, k])) \
                       + opfobj.M_if_qY.dot(opfobj.pv_inc_matrix.dot(q_pv_dist[:, k]))

        residual_v_max = voltage_expr - voltage_max_lims
        residual_v_min = voltage_expr - voltage_min_lims

        residual_i_ij_max = current_expr - max_line_ratings

        residual_i_ij_min = current_expr - min_line_ratings

        residual_v_max_viol_idx[k] = any(residual_v_max <= 0)
        residual_v_min_viol_idx[k] = any(residual_v_min >= 0)
        residual_ij_max_viol_idx[k] = any(residual_i_ij_max <= 0)
        residual_ij_min_viol_idx[k] = any(residual_i_ij_min >= 0)

        if ((abs(residual_bal_P[k + 1]) < tolerance)
                and (abs(residual_bal_Q[k + 1]) < tolerance)
                and (residual_ij_max_viol_idx[k])
                and (residual_ij_min_viol_idx[k])
                and (residual_v_max_viol_idx[k])
                and (residual_v_min_viol_idx[k])):
            term = 1

        k += 1
    stop_distributed = timeit.default_timer()
    distributed_times = stop_distributed - start_distributed
    iteration_times = k


