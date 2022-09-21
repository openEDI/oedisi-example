
# -*- coding: utf-8 -*-
# import helics as h
import opendssdirect as dss
import numpy as np
import csv
import time
from time import strptime
from scipy.sparse import coo_matrix
import os
import random
import math
import logging
import json

from pydantic import BaseModel

from dss_functions import (
    snapshot_run, parse_Ymatrix, get_loads, get_pvSystems, get_Generator,
    get_capacitors, get_voltages, get_y_matrix_file, get_vnom, get_vnom2
)
import dss_functions

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)

def permutation(from_list, to_list):
    """
    Create permutation representing change in from_list to to_list

    Specifically, if `permute = permutation(from_list, to_list)`,
    then `permute[i] = j` means that `from_list[i] = to_list[j]`.

    This also means that `to_list[permute] == from_list`, so you
    can convert from indices under to_list to indices under from_list.

    You may view the permutation as a function from `from_list` to `to_list`.
    """
    #return [to_list.find(v) for v in enumerate(from_list)]
    index_map = {v: i for i, v in enumerate(to_list)}
    return [index_map[v] for v in from_list]



def check_node_order(l1, l2):
    logger.debug('check order ' + str(l1 == l2))


class FeederConfig(BaseModel):
    name: str
    feeder_file: str
    start_date: str
    number_of_timesteps: float
    run_freq_sec: float = 15*60
    start_time_index: int = 0
    topology_output: str = "topology.json"


class FeederSimulator(object):
    """ A simple class that handles publishing the solar forecast
    """

    def __init__(self, config):
        """ Create a ``FeederSimulator`` object

        """
        self._feeder_file = config.feeder_file

        self._circuit=None
        self._AllNodeNames=None
        self._source_indexes=None
        # self._capacitors=[]
        self._capNames = []
        self._regNames = []

        # timegm(strptime('2019-07-23 14:50:00 GMT', '%Y-%m-%d %H:%M:%S %Z'))
        self._start_time = int(time.mktime(strptime(config.start_date, '%Y-%m-%d %H:%M:%S')))
        self._run_freq_sec = config.run_freq_sec
        self._simulation_step = config.start_time_index
        self._simulation_time_step = self._start_time
        self._number_of_timesteps = config.number_of_timesteps
        self._vmult = 0.001

        self._nodes_index = []
        self._name_index_dict = {}

        self.load_feeder()
        #self.create_measurement_lists()

    def create_measurement_lists(self,
            percent_voltage=75,
            percent_real=75,
            percent_reactive=75,
            voltage_seed=1,
            real_seed=2,
            reactive_seed=3
        ):

        random.seed(voltage_seed)
        voltage_subset = random.sample(self._AllNodeNames,math.floor(len(self._AllNodeNames)*float(percent_voltage)/100))
        with open('voltage_ids.json','w') as fp:
            json.dump(voltage_subset,fp,indent=4)

        random.seed(real_seed)
        real_subset = random.sample(self._AllNodeNames,math.floor(len(self._AllNodeNames)*float(percent_real)/100))
        with open('real_ids.json','w') as fp:
            json.dump(real_subset,fp,indent=4)

        random.seed(reactive_seed)
        reactive_subset = random.sample(self._AllNodeNames,math.floor(len(self._AllNodeNames)*float(percent_voltage)/100))
        with open('reactive_ids.json','w') as fp:
            json.dump(reactive_subset,fp,indent=4)
    def setup_player(self):
        dss.run_command('set mode=yearly loadmult=1 number=1 stepsize=1m ')

    def snapshot_run(self):
        snapshot_run(dss)

    def get_circuit_name(self):
        return self._circuit.Name()

    def get_source_indices(self):
        return self._source_indexes

    def get_node_names(self):
        return self._AllNodeNames


    def load_feeder(self):
        result = dss.run_command("redirect " + self._feeder_file)
        if not result == '':
            raise ValueError("Feeder not loaded: "+result)
        self._circuit = dss.Circuit
        self._AllNodeNames = self._circuit.YNodeOrder()
        self._node_number = len(self._AllNodeNames)
        self._nodes_index = [self._AllNodeNames.index(ii) for ii in self._AllNodeNames]
        self._name_index_dict = {ii: self._AllNodeNames.index(ii) for ii in self._AllNodeNames}

        self._source_indexes = []
        for Source in dss.Vsources.AllNames():
            self._circuit.SetActiveElement('Vsource.' + Source)
            Bus = dss.CktElement.BusNames()[0].upper()
            for phase in range(1, dss.CktElement.NumPhases() + 1):
                self._source_indexes.append(self._AllNodeNames.index(Bus.upper() + '.' + str(phase)))


        self.setup_vbase()

    def get_y_matrix(self):
        get_y_matrix_file(dss)
        Ymatrix = parse_Ymatrix('base_ysparse.csv', self._node_number)
        new_order = self._circuit.YNodeOrder()
        permute = np.array(permutation(new_order, self._AllNodeNames))
        #inv_permute = np.array(permutation(self._AllNodeNames, new_order))
        return coo_matrix((Ymatrix.data, (permute[Ymatrix.row], permute[Ymatrix.col])), shape=Ymatrix.shape)


    def setup_vbase(self):
        self._Vbase_allnode = np.zeros((self._node_number), dtype=np.complex_)
        self._Vbase_allnode_dict = {}
        for ii, node in enumerate(self._AllNodeNames):
            self._circuit.SetActiveBus(node)
            self._Vbase_allnode[ii] = dss.Bus.kVBase() * 1000
            self._Vbase_allnode_dict[node] = self._Vbase_allnode[ii]

    def get_G_H(self, Y11_inv):
        Vnom = self.get_vnom()
        # ys=Y11
        # R = np.linalg.inv(ys).real
        # X = np.linalg.inv(ys).imag
        R=Y11_inv.real
        X=Y11_inv.imag
        G = (R * np.diag(np.cos(np.angle(Vnom)) / abs(Vnom)) - X * np.diag(np.sin(np.angle(Vnom)) / Vnom)).real
        H = (X * np.diag(np.cos(np.angle(Vnom)) / abs(Vnom)) - R * np.diag(np.sin(np.angle(Vnom)) / Vnom)).real
        return Vnom, G, H

    def get_vnom2(self):
        _Vnom, Vnom_dict = get_vnom2(dss)
        Vnom = np.zeros((len(self._AllNodeNames)), dtype=np.complex_)
        for voltage_name in Vnom_dict.keys():
            Vnom[self._name_index_dict[voltage_name]] = Vnom_dict[voltage_name]
        # Vnom(1: 3) = [];
        Vnom = np.concatenate((Vnom[:self._source_indexes[0]], Vnom[self._source_indexes[-1] + 1:]))
        return Vnom

    def get_vnom(self):
        _Vnom, Vnom_dict = get_vnom(dss)
        Vnom = np.zeros((len(self._AllNodeNames)), dtype=np.complex_)
        # print([name_voltage_dict.keys()][:5])
        for voltage_name in Vnom_dict.keys():
            Vnom[self._name_index_dict[voltage_name]] = Vnom_dict[voltage_name]
        # Vnom(1: 3) = [];
        logger.debug(Vnom[self._source_indexes[0]:self._source_indexes[-1]])
        Vnom = np.concatenate((Vnom[:self._source_indexes[0]], Vnom[self._source_indexes[-1] + 1:]))
        Vnom = np.abs(Vnom)
        return Vnom

    def get_PQs_load(self,static=False):
        num_nodes = len(self._name_index_dict.keys())

        PQ_names= ['' for i in range(num_nodes)]
        PQ_types= ['' for i in range(num_nodes)]
        PQ_load = np.zeros((num_nodes), dtype=np.complex_)
        for ld in get_loads(dss,self._circuit):
            self._circuit.SetActiveElement('Load.' + ld["name"])
            for ii in range(len(ld['phases'])):
                name = ld['bus1'].upper() + '.' + ld['phases'][ii]
                index = self._name_index_dict[name]
                if static:
                    power = complex(ld['kW'],ld['kVar'])
                    PQ_load[index] += power/len(ld['phases'])
                else:
                    power = dss.CktElement.Powers()
                    PQ_load[index] += np.complex(power[2 * ii], power[2 * ii + 1])
                PQ_names[index] = name
                PQ_types[index] = 'Load'

        return PQ_load,PQ_names,PQ_types

    def get_PQs_pv(self,static=False):
        num_nodes = len(self._name_index_dict.keys())

        PQ_names= ['' for i in range(num_nodes)]
        PQ_types= ['' for i in range(num_nodes)]
        PQ_PV = np.zeros((num_nodes), dtype=np.complex_)
        for PV in get_pvSystems(dss):
            bus = PV["bus"].split('.')
            if len(bus) == 1:
                bus = bus + ['1', '2', '3']
            self._circuit.SetActiveElement('PVSystem.'+PV["name"])
            for ii in range(len(bus) - 1):
                name = bus[0].upper() + '.' + bus[ii + 1]
                index = self._name_index_dict[name]
                if static:
                    power = complex(-1*PV['kW'],-1*PV['kVar']) #-1 because injecting
                    PQ_PV[index] += power/(len(bus)-1)
                else:
                    power = dss.CktElement.Powers()
                    PQ_PV[index] += np.complex(power[2 * ii], power[2 * ii + 1])
                PQ_names[index] = name
                PQ_types[index] = 'PVSystem'
        return PQ_PV,PQ_names,PQ_types

    def get_PQs_gen(self,static=False):
        num_nodes = len(self._name_index_dict.keys())

        PQ_names= ['' for i in range(num_nodes)]
        PQ_types= ['' for i in range(num_nodes)]
        PQ_gen = np.zeros((num_nodes), dtype=np.complex_)
        for PV in get_Generator(dss):
            bus = PV["bus"]
            self._circuit.SetActiveElement('Generator.'+PV["name"])
            for ii in range(len(bus) - 1):
                name = bus[0].upper() + '.' + bus[ii + 1]
                index = self._name_index_dict[name]
                if static:
                    power = complex(-1*PV['kW'],-1*PV['kVar']) #-1 because injecting
                    PQ_gen[index] += power/(len(bus)-1)
                else:
                    power = dss.CktElement.Powers()
                    PQ_gen[index] += np.complex(power[2 * ii], power[2 * ii + 1])
                PQ_names[index] = name
                PQ_types[index] = 'Generator'
        return PQ_gen,PQ_names,PQ_types

    def get_PQs_cap(self,static=False):
        num_nodes = len(self._name_index_dict.keys())

        PQ_names= ['' for i in range(num_nodes)]
        PQ_types= ['' for i in range(num_nodes)]
        PQ_cap = np.zeros((num_nodes), dtype=np.complex_)
        for cap in get_capacitors(dss):
            for ii in range(cap["numPhases"]):
                name = cap["busname"].upper() + '.' + cap["busphase"][ii]
                index = self._name_index_dict[name]
                if static:
                    power = complex(0,-1*cap['kVar']) #-1 because it's injected into the grid
                    PQ_cap[index] += power/cap["numPhases"]
                else:
                    PQ_cap[index] = np.complex(0,cap["power"][2 * ii + 1])
                PQ_names[index] = name
                PQ_types[index] = 'Capacitor'

        return PQ_cap,PQ_names,PQ_types




    def get_loads(self):
        loads = get_loads(dss, self._circuit)
        self._load_power = np.zeros((len(self._AllNodeNames)), dtype=np.complex_)
        load_names = []
        load_powers = []
        load = loads[0]
        for load in loads:
            for phase in load['phases']:
                self._load_power[
                    self._name_index_dict[load['bus1'].upper() + '.' + phase]
                ] = complex(load['power'][0], load['power'][1])
                load_names.append(load['bus1'].upper() + '.' + phase)
                load_powers.append(complex(load['power'][0], load['power'][1]))
        return self._load_power, load_names, load_powers

    def get_load_sizes(self):
        return dss_functions.get_load_sizes(dss,self._loads)

    def get_voltages_actual(self):
        '''

        :return voltages in actual values:
        '''
        _, name_voltage_dict = get_voltages(self._circuit)
        res_feeder_voltages = np.zeros((len(self._AllNodeNames)), dtype=np.complex_)
        for voltage_name in name_voltage_dict.keys():
            res_feeder_voltages[self._name_index_dict[voltage_name]] = name_voltage_dict[voltage_name]

        return res_feeder_voltages

    def get_voltages(self):
        '''

        :return per unit voltages:
        '''
        res_feeder_voltages = np.abs(self.get_voltages_complex())
        return res_feeder_voltages

    def get_voltages_complex(self):
        '''

        :return per unit voltages:
        '''
        #voltages = np.array(self._circuit.AllBusVolts())
        #voltages = voltages[::2] + 1j * voltages[1::2]
        #assert len(voltages) == len(self._AllNodeNames)
        #return voltages

        _, name_voltage_dict = get_voltages(self._circuit)
        # print('Publish load ' + str(temp_feeder_voltages.real[0]))
        # set_load(dss, data_df['load'], current_time, loads)
        # feeder_voltages = [0j] * len(self._AllNodeNames)
        res_feeder_voltages = np.zeros((len(self._AllNodeNames)), dtype=np.complex_)
        # print([name_voltage_dict.keys()][:5])
        temp_array_pu = np.zeros((len(self._AllNodeNames)), dtype=np.complex_)
        for voltage_name in name_voltage_dict.keys():

            self._circuit.SetActiveBus(voltage_name)
            # temp1 = dss.Bus.puVmagAngle()
            temp = dss.Bus.PuVoltage()
            temp = complex(temp[0], temp[1])
            temp_array_pu[self._name_index_dict[voltage_name]] = temp
            res_feeder_voltages[self._name_index_dict[voltage_name]] = name_voltage_dict[voltage_name] / (self._Vbase_allnode_dict[voltage_name])

        return res_feeder_voltages

    def set_pv_list(self, list_p, list_q):
        """
        List of generator p and qs that need to match the order of gen names
        :param list_p:
        :param list_q:
        :return:
        """
        for i, name in enumerate(self._pv_names):
            dss.run_command('edit PVSystem.' + name + ' kW=' + str(list_p[i]) + ' kVar=' + str(list_q[i]))

    def set_load_list(self, list_p, list_q):
        """
        List of load p and qs that need to match the order of load names
        :param list_p:
        :param list_q:
        :return:
        """
        for i, name in enumerate(self._load_names):
            # print('edit Load.' + str(load["name"]) + ' kW=' + str(loadshape_p) + ' kVar=' + str(loadshape_q))
            dss.run_command('edit Load.' + name + ' kW=' + str(list_p[i]) + ' kVar=' + str(list_q[i]))

    def set_load_time_series(self, df):
        """
        Timeseries dataframe of correct frequency with load data
        :param df:
        :return:
        """
        df_for_current_time = df.loc[self._simulation_step]
        for load in self._loads:
            loadshape_p = df_for_current_time[load["name"]]
            loadshape_q = df_for_current_time[load["name"]+'q']
            # print('edit Load.' + str(load["name"]) + ' kW=' + str(loadshape_p) + ' kVar=' + str(loadshape_q))
            dss.run_command('edit Load.' + load["name"] + ' kW=' + str(loadshape_p) + ' kVar=' + str(loadshape_q))

    def set_pv_time_series(self, df):
        df_for_current_time = df.loc[self._simulation_step]
        for element in self._pvs:
            name_ = element["name"]
            loadshape_p = df_for_current_time[name_]
            loadshape_q = df_for_current_time[name_ + 'q']
            # print('edit PVSystem.' + name_ + ' kW=' + str(loadshape_p) + ' kVar=' + str(loadshape_q))
            dss.run_command('edit PVSystem.' + name_ + ' kW=' + str(loadshape_p) + ' kVar=' + str(loadshape_q))

    def set_gen_time_series(self, df):
        df_for_current_time = df.loc[self._simulation_step]
        for element in self._gens:
            name_ = element["name"]
            loadshape_p = df_for_current_time[name_]
            loadshape_q = df_for_current_time[name_ + 'q']
            # print('edit Generator.' + name_ + ' kW=' + str(loadshape_p) + ' kVar=' + str(loadshape_q))
            dss.run_command('edit Generator.' + name_ + ' kW=' + str(loadshape_p) + ' kVar=' + str(loadshape_q))

    def set_load_pq_timeseries(self, loadshape_p):
        # dss.run_command("show voltages LN nodes")
        # exit(0)
        df_for_current_time = loadshape_p.loc[self._simulation_step]
        for load in self._loads:
            name = load["name"]
            num_phase = load["numPhases"]
            dss.run_command('edit Load.' + str(name) + ' kW=' + str(float(load['kW']) * df_for_current_time[name].real)
                            + ' kVar=' + str(float(load['kVar']) * df_for_current_time[name].imag))


    def set_gen_pq_timeseries(self, loadshape_p):
        df_for_current_time = loadshape_p.loc[self._simulation_step]
        for gen in self._gens:
            name = gen["name"]
            dss.run_command('edit Generator.' + str(name) + ' kW=' + str(float(gen['kW']) * df_for_current_time[name].real)
                            + ' kVar=' + str(float(gen['kVar']) * df_for_current_time[name].imag))

    def set_pv_pq_timeseries(self, loadshape_p):
        df_for_current_time = loadshape_p.loc[self._simulation_step]
        for pvs in self._pvs:
            name = pvs["name"]
            dss.run_command('edit PVSystem.' + str(name) + ' kW=' + str(
                float(pvs['kW']) * df_for_current_time[name].real)
                            + ' kVar=' + str(float(pvs['kVar']) * df_for_current_time[name].imag))

    def set_load_pq(self, loadshape_p=1. , loadshape_q=1.):
        #     str(float(load['kW']) * loadshape)
        for load in self._loads:
            dss.run_command('edit Load.' + str(load["name"]) + ' kW=' + str(float(load['kW']) * loadshape_p) + ' kVar=' + str(float(load['kVar']) * loadshape_q))

    def set_pv_pq(self, loadshape_p=1., loadshape_q=1.):
        for pv in self._pvs:
            dss.run_command('edit PVSystem.' + str(pv["name"]) + ' kW=' + str(float(pv['kW']) * loadshape_p) + ' kvar=' + str(float(pv['kVar']) * loadshape_q))

    def set_gen_pq(self, loadshape_p=1., loadshape_q=1.):
        for gen in self._gens:
            dss.run_command('edit Generator.' + str(gen["name"]) + ' kW=' + str(float(gen['kW']) * loadshape_p) + ' kvar=' + str(float(gen['kVar']) * loadshape_q))

    def run_command(self, cmd):
        dss.run_command(cmd)

    def solve(self,hour,second):
        dss.run_command(f'set mode=yearly loadmult=1 number=1 hour={hour} sec={second} stepsize={self._simulation_time_step} ')
        dss.run_command('solve')


    def run_next(self):
        # snapshot_run(dss)
        dss.run_command('solve')
        self._simulation_step += 1
        self._simulation_time_step += self._run_freq_sec

    def run_next_control(self, load_df, pv_df, gen_df):
        # snapshot_run(dss)
        logger.debug(type(load_df) )
        if type(load_df) == complex:
            self.set_load_pq(1., 1.)
            self.set_pv_pq(1., 1.)
            self.set_gen_pq(1., 1.)
        else:
            self.set_load_time_series(load_df)
            self.set_pv_time_series(pv_df)
            self.set_gen_time_series(gen_df)

        self.run_next()
