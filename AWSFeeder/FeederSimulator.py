
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
import math
import random
import json
from scipy.sparse import coo_matrix
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


class FeederConfig(BaseModel):
    name: str
    smartds_region: str
    smartds_feeder: str
    smartds_scenario: str
    smartds_year: str
    start_date: str
    increment_value: int # increment in seconds
    number_of_timesteps: int


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
        self._number_of_timesteps = config.number_of_timesteps
        self._increment_value = config.increment_value

        self._feeder_file = None
        self._circuit=None
        self._AllNodeNames=None
        self._source_indexes=None

        self._start_time = int(time.mktime(strptime(config.start_date, '%Y-%m-%d %H:%M:%S')))
        self._vmult = 0.001

        self._nodes_index = []
        self._name_index_dict = {}

        self.download_data()
        self.load_feeder()
        self.create_measurement_lists()


    def download_data(self):

        bucket_name = 'oedi-data-lake'

        #Equivalent to --no-sign-request
        s3_resource = boto3.resource('s3',config=Config(signature_version=UNSIGNED))
        bucket = s3_resource.Bucket(bucket_name)
        opendss_location = f'SMART-DS/v1.0/{self._smartds_year}/SFO/{self._smartds_region}/scenarios/{self._smartds_scenario}/opendss/{self._smartds_feeder}'
        profile_location =f'SMART-DS/v1.0/{self._smartds_year}/SFO/{self._smartds_region}' 

        self._feeder_file = os.path.join('opendss','Master.dss')
        self._simulation_time_step = '15m'
        for obj in bucket.objects.filter(Prefix=opendss_location):
            output_location = os.path.join('opendss',obj.key.replace(opendss_location,''))
            if not os.path.exists(os.path.dirname(output_location)):
                os.makedirs(os.path.dirname(output_location))
            bucket.download_file(obj.key,output_location)

        modified_loadshapes = ''
        all_profiles = set()
        if not os.path.exists(os.path.join('opendss','profiles')):
            os.makedirs(os.path.join('opendss','profiles'))
        with open(os.path.join('opendss','LoadShapes.dss'),'r') as fp_loadshapes:
            for row in fp_loadshapes.readlines():
                new_row = row.replace('../','')
                for token in new_row.split(' '):
                    if token.startswith('(file='):
                        location = token.split('=')[1].strip().strip(')')
                        all_profiles.add(location)
                modified_loadshapes=modified_loadshapes+new_row
        with open(os.path.join('opendss','LoadShapes.dss'),'w') as fp_loadshapes:
            fp_loadshapes.write(modified_loadshapes)
        for profile in all_profiles:
            s3_location = f'{profile_location}/{profile}'
            bucket.download_file(s3_location,os.path.join('opendss',profile))

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
        Ymatrix_old, Ymatrix = parse_Ymatrix('base_ysparse.csv', self._node_number)
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

    def get_PQs(self):
        num_nodes = len(self._name_index_dict.keys())

        PQ_load = np.zeros((num_nodes), dtype=np.complex_)
        for ld in get_loads(dss,self._circuit):
            for ii in range(len(ld['phases'])):
                name = ld['bus1'] + '.' + ld['phases'][ii]
                index = self._name_index_dict[name.upper()]
                self._circuit.SetActiveElement('Load.' + ld["name"])
                power = dss.CktElement.Powers()
                PQ_load[index] += np.complex(power[2 * ii], power[2 * ii + 1])

        PQ_PV = np.zeros((num_nodes), dtype=np.complex_)
        for PV in get_pvSystems(dss):
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
        for PV in get_Generator(dss):
            bus = PV["bus"]
            self._circuit.SetActiveElement('Generator.'+PV["name"])
            power = dss.CktElement.Powers()
            for ii in range(len(bus) - 1):
                index = self._name_index_dict[(bus[0] + '.' + bus[ii + 1]).upper()]
                PQ_gen_all[index] += np.complex(power[2 * ii], power[2 * ii + 1])
                PQ_gen[index] += np.complex(power[2 * ii], power[2 * ii + 1])

        Qcap = [0] * num_nodes
        for cap in get_capacitors(dss):
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


    def run_command(self, cmd):
        dss.run_command(cmd)

    def solve(self,hour,second):
        dss.run_command(f'set mode=yearly loadmult=1 number=1 hour={hour} sec={second} stepsize={self._simulation_time_step} ')
        dss.run_command('solve')

