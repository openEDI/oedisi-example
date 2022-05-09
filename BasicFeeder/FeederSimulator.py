
# -*- coding: utf-8 -*-
# import helics as h
import opendssdirect as dss
import numpy as np
import csv
import time
import sys
from time import strptime
import os
import logging
import boto3
from botocore import UNSIGNED
from botocore.config import Config

from pydantic import BaseModel

from dss_functions import (
    snapshot_run, parse_Ymatrix, get_loads, get_pvSystems, get_Generator,
    get_capacitors, get_voltages, get_y_matrix_file, get_vnom, get_vnom2
)
import dss_functions

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)

def check_node_order(l1, l2):
    print('check order ' + str(l1 == l2))


class FeederConfig(BaseModel):
    smartds_region: str
    smartds_feeder: str
    smartds_scenario: str
    smartds_year: str
    start_date: str
    end_date: str
    increment_value: int # increment in seconds


class FeederSimulator(object):
    """ A simple class that handles publishing the solar forecast
    """

    def __init__(self, config):
        """ Create a ``FeederSimulator`` object

        """
        self._smartds_region = config.smartds_region
        self._smartds_feeder = config.smartds_feeder
        self._smartds_scenario = config.smartds_scenario
        self._smartds_year = config.smartds_year
        self._start_date = config.start_date
        self._end_date = config.end_date
        self._feeder_file = None

        self._circuit=None
        self._AllNodeNames=None
        self._source_indexes=None
        # self._capacitors=[]
        self._capNames = []
        self._regNames = []

        # timegm(strptime('2019-07-23 14:50:00 GMT', '%Y-%m-%d %H:%M:%S %Z'))
        self._start_time = int(time.mktime(strptime(config.start_date, '%Y-%m-%d %H:%M:%S')))
        #self._run_freq_sec = config.run_freq_sec
        #self._simulation_step = config.start_time_index
        #self._simulation_time_step = self._start_time
        self._vmult = 0.001

        self._nodes_index = []
        self._name_index_dict = {}

        self.download_data()
        self.load_feeder()


    def download_data(self):

        bucket_name = 'oedi-data-lake'
        #Equivalent to --no-sign-request
        s3_resource = boto3.resource('s3',config=Config(signature_version=UNSIGNED))
        bucket = s3_resource.Bucket(bucket_name)
        opendss_location = f'SMART-DS/v1.0/{self._smartds_year}/SFO/{self._smartds_region}/scenarios/{self._smartds_scenario}/opendss/{self._smartds_feeder}'
        profile_location =f'SMART-DS/v1.0/{self._smartds_year}/SFO/{self._smartds_region}/profiles' 

        self._feeder_file = os.path.join('opendss','Master.dss')
        self._simulation_time_step = '15m'
        for obj in bucket.objects.filter(Prefix=opendss_location):
            output_location = os.path.join('opendss',obj.key.replace(opendss_location,''))
            if not os.path.exists(os.path.dirname(output_location)):
                os.makedirs(os.path.dirname(output_location))
            bucket.download_file(obj.key,output_location)


#        for obj in bucket.objects.filter(Prefix=profile_location):
#            output_location = os.path.join('profiles',obj.key.replace(opendss_location,''))
#            if not os.path.exists(os.path.dirname(output_location)):
#                os.makedirs(os.path.dirname(output_location))
#            bucket.download_file(obj.key,output_location)

    def setup_player(self):
        dss.run_command(f'set mode=yearly loadmult=1 number=1 stepsize={self._simulation_time_step} ')

    def snapshot_run(self):
        snapshot_run(dss)

    def get_circuit_name(self):
        return self._circuit.Name()

    def get_source_indices(self):
        return self._source_indexes

    def get_node_names(self):
        return self._AllNodeNames

    def load_feeder(self):
        dss.run_command("redirect " + self._feeder_file)


    def get_y_matrix(self):
        get_y_matrix_file(dss)
        self._AllNodeNames = self._circuit.YNodeOrder()
        self._node_number = len(self._AllNodeNames)
        AllNodeNames = []
        with open(os.path.join('.', 'base_nodelist.csv')) as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            for row in reader:
                AllNodeNames.append(row[0])
        check_node_order(self._AllNodeNames,AllNodeNames)
        self._nodes_index = [self._AllNodeNames.index(ii) for ii in self._AllNodeNames]
        self._name_index_dict = {ii: self._AllNodeNames.index(ii) for ii in self._AllNodeNames}
        print(self._AllNodeNames)

        ## For y-matrix pu
        self.setup_vbase()

        self._expect_max_name_load_dict = {}
        self._expect_max_load_power = np.zeros((len(self._AllNodeNames)), dtype=np.complex_)
        for load in self._loads:
            for phase in load['phases']:
                # print(load['bus1'].upper()+'.'+phase, name_index_dict[load['bus1'].upper()+'.'+phase])
                self._expect_max_name_load_dict[load['bus1'].upper() + '.' + phase] = complex(load['kW'],load['kVar'])
                self._expect_max_load_power[self._name_index_dict[load['bus1'].upper() + '.' + phase]] = complex(load['kW'], load['kVar'])

        self._source_indexes = []
        for Source in dss.Vsources.AllNames():
            self._circuit.SetActiveElement('Vsource.' + Source)
            Bus = dss.CktElement.BusNames()[0].upper()
            for phase in range(1, dss.CktElement.NumPhases() + 1):
                self._source_indexes.append(self._AllNodeNames.index(Bus.upper() + '.' + str(phase)))
        logger.debug("Voltage Sources")
        logger.debug(self._source_indexes)

        Ymatrix = parse_Ymatrix('base_ysparse.csv', self._node_number)

        self.snapshot_run()
        self.setup_vbase()
        temp_AllNodeNames = self._circuit.YNodeOrder()
        check_node_order(temp_AllNodeNames, self._AllNodeNames)

        return Ymatrix

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
        print(Vnom[self._source_indexes[0]:self._source_indexes[-1]])
        Vnom = np.concatenate((Vnom[:self._source_indexes[0]], Vnom[self._source_indexes[-1] + 1:]))
        Vnom = np.abs(Vnom)
        return Vnom

    def get_PQs(self):
        num_nodes = len(self._name_index_dict.keys())

        PQ_load = np.zeros((num_nodes), dtype=np.complex_)
        for ld in self._loads:
            for ii in range(len(ld['phases'])):
                name = ld['bus1'] + '.' + ld['phases'][ii]
                index = self._name_index_dict[name.upper()]
                self._circuit.SetActiveElement('Load.' + ld["name"])
                power = dss.CktElement.Powers()
                PQ_load[index] += np.complex(power[2 * ii], power[2 * ii + 1])

        PQ_PV = np.zeros((num_nodes), dtype=np.complex_)
        for PV in self._pvs:
            bus = PV["bus"].split('.')
            if len(bus) == 1:
                bus = bus + ['1', '2', '3']
            self._circuit.SetActiveElement('PVSystem.'+PV["name"])
            power = dss.CktElement.Powers()
            for ii in range(len(bus) - 1):
                index = self._name_index_dict[(bus[0] + '.' + bus[ii + 1]).upper()]
                PQ_PV[index] += np.complex(power[2 * ii], power[2 * ii + 1])

        PQ_gen = np.zeros((num_nodes), dtype=np.complex_)
        PQ_gen_all = np.zeros((num_nodes), dtype=np.complex_)
        for PV in self._gens:
            bus = PV["bus"]
            self._circuit.SetActiveElement('Generator.'+PV["name"])
            power = dss.CktElement.Powers()
            for ii in range(len(bus) - 1):
                index = self._name_index_dict[(bus[0] + '.' + bus[ii + 1]).upper()]
                PQ_gen_all[index] += np.complex(power[2 * ii], power[2 * ii + 1])
                PQ_gen[index] += np.complex(power[2 * ii], power[2 * ii + 1])

        Qcap = [0] * num_nodes
        for cap in self._caps:
            for ii in range(cap["numPhases"]):
                index = self._name_index_dict[cap["busname"].upper() + '.' + cap["busphase"][ii]]
                Qcap[index] = -cap["power"][2 * ii - 1]

        return PQ_load + PQ_PV + PQ_gen + 1j * np.array(Qcap)  # power injection


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

        _, name_voltage_dict = get_voltages(self._circuit)
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

    def solve(self):
        # snapshot_run(dss)
        dss.run_command('solve')

    def run_next(self):
        # snapshot_run(dss)
        dss.run_command('solve')
        self._simulation_step += 1
        self._simulation_time_step += self._run_freq_sec

    def run_next_control(self, load_df, pv_df, gen_df):
        # snapshot_run(dss)
        print(type(load_df) )
        if type(load_df) == complex:
            self.set_load_pq(1., 1.)
            self.set_pv_pq(1., 1.)
            self.set_gen_pq(1., 1.)
        else:
            self.set_load_time_series(load_df)
            self.set_pv_time_series(pv_df)
            self.set_gen_time_series(gen_df)

        self.run_next()
