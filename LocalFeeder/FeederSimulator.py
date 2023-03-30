# -*- coding: utf-8 -*-
# import helics as h
from typing import Any, List
import opendssdirect as dss
import numpy as np
import time
from time import strptime
from scipy.sparse import coo_matrix
import os
import random
import math
import logging
import json
import boto3
from botocore import UNSIGNED
from botocore.config import Config

from pydantic import BaseModel
from scipy.sparse import csc_matrix
from enum import Enum

from dss_functions import (
    get_loads,
    get_pvSystems,
    get_Generator,
    get_capacitors,
    get_voltages,
    get_vnom,
    get_vnom2,
)
import xarray as xr

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
    # return [to_list.find(v) for v in enumerate(from_list)]
    index_map = {v: i for i, v in enumerate(to_list)}
    return [index_map[v] for v in from_list]


def check_node_order(l1, l2):
    logger.debug("check order " + str(l1 == l2))


class FeederConfig(BaseModel):
    name: str
    use_smartds: bool = False
    profile_location: str
    opendss_location: str
    sensor_location: str = ""
    start_date: str
    number_of_timesteps: float
    run_freq_sec: float = 15 * 60
    start_time_index: int = 0
    topology_output: str = "topology.json"
    use_sparse_admittance = False


class Command(BaseModel):
    obj_name: str
    obj_property: str
    val: Any


class CommandList(BaseModel):
    __root__: List[Command]


class OpenDSSState(Enum):
    UNLOADED = 1
    LOADED = 2
    SNAPSHOT_RUN = 3
    SOLVE_AT_TIME = 4
    DISABLED_RUN = 5
    DISABLED_SOLVE = 6


class FeederSimulator(object):
    """ A simple class that handles publishing the solar forecast
    """

    def __init__(self, config: FeederConfig):
        """ Create a ``FeederSimulator`` object

        """
        self._state = OpenDSSState.UNLOADED
        self._feeder_file = None
        self._simulation_time_step = None
        self._opendss_location = config.opendss_location
        self._profile_location = config.profile_location
        self._sensor_location = config.sensor_location
        self._use_smartds = config.use_smartds

        self._circuit = None
        self._AllNodeNames = None
        self._source_indexes = None
        # self._capacitors=[]
        self._capNames = []
        self._regNames = []

        # timegm(strptime('2019-07-23 14:50:00 GMT', '%Y-%m-%d %H:%M:%S %Z'))
        self._start_time = int(
            time.mktime(strptime(config.start_date, "%Y-%m-%d %H:%M:%S"))
        )
        self._run_freq_sec = config.run_freq_sec
        self._simulation_step = config.start_time_index
        self._number_of_timesteps = config.number_of_timesteps
        self._vmult = 0.001

        self._nodes_index = []
        self._name_index_dict = {}

        self._simulation_time_step = "15m"
        if self._use_smartds:
            self._feeder_file = os.path.join("opendss", "Master.dss")
            if not os.path.isfile(os.path.join("opendss", "Master.dss")):
                self.download_data("oedi-data-lake")
            self.load_feeder()
            self.create_measurement_lists()
        else:
            self._feeder_file = os.path.join("opendss", "master.dss")
            if not os.path.isfile(os.path.join("opendss", "master.dss")):
                self.download_data("gadal")
            self.load_feeder()

        self.snapshot_run()
        assert self._state == OpenDSSState.SNAPSHOT_RUN, f"{self._state}"

    def snapshot_run(self):
        assert self._state != OpenDSSState.UNLOADED, f"{self._state}"
        dss.run_command("Batchedit Load..* enabled=yes")
        dss.run_command("Batchedit Vsource..* enabled=yes")
        dss.run_command("Batchedit Isource..* enabled=yes")
        dss.run_command("Batchedit Generator..* enabled=yes")
        dss.run_command("Batchedit PVsystem..* enabled=yes")
        dss.run_command("Batchedit Capacitor..* enabled=yes")
        dss.run_command("Batchedit Storage..* enabled=no")
        dss.run_command("CalcVoltageBases")
        dss.run_command("solve mode=snapshot")
        self._state = OpenDSSState.SNAPSHOT_RUN

    def download_data(self, bucket_name, update_loadshape_location=False):
        logging.info(f"Downloading from bucket {bucket_name}")
        # Equivalent to --no-sign-request
        s3_resource = boto3.resource("s3", config=Config(signature_version=UNSIGNED))
        bucket = s3_resource.Bucket(bucket_name)
        opendss_location = self._opendss_location
        profile_location = self._profile_location
        sensor_location = self._sensor_location

        for obj in bucket.objects.filter(Prefix=opendss_location):
            output_location = os.path.join(
                "opendss", obj.key.replace(opendss_location, "").strip("/")
            )
            os.makedirs(os.path.dirname(output_location), exist_ok=True)
            bucket.download_file(obj.key, output_location)

        modified_loadshapes = ""
        os.makedirs(os.path.join("profiles"), exist_ok=True)
        if update_loadshape_location:
            all_profiles = set()
            with open(os.path.join("opendss", "LoadShapes.dss"), "r") as fp_loadshapes:
                for row in fp_loadshapes.readlines():
                    new_row = row.replace("../", "")
                    new_row = new_row.replace("file=", "file=../")
                    for token in new_row.split(" "):
                        if token.startswith("(file="):
                            location = (
                                token.split("=../profiles/")[1].strip().strip(")")
                            )
                            all_profiles.add(location)
                    modified_loadshapes = modified_loadshapes + new_row
            with open(os.path.join("opendss", "LoadShapes.dss"), "w") as fp_loadshapes:
                fp_loadshapes.write(modified_loadshapes)
            for profile in all_profiles:
                s3_location = f"{profile_location}/{profile}"
                bucket.download_file(s3_location, os.path.join("profiles", profile))

        else:
            for obj in bucket.objects.filter(Prefix=profile_location):
                output_location = os.path.join(
                    "profiles", obj.key.replace(profile_location, "").strip("/")
                )
                os.makedirs(os.path.dirname(output_location), exist_ok=True)
                bucket.download_file(obj.key, output_location)

        if sensor_location != "":
            output_location = os.path.join("sensors", os.path.basename(sensor_location))
            if not os.path.exists(os.path.dirname(output_location)):
                os.makedirs(os.path.dirname(output_location))
            bucket.download_file(sensor_location, output_location)

    def create_measurement_lists(
        self,
        percent_voltage=75,
        percent_real=75,
        percent_reactive=75,
        voltage_seed=1,
        real_seed=2,
        reactive_seed=3,
    ):
        random.seed(voltage_seed)
        os.makedirs("sensors", exist_ok=True)
        voltage_subset = random.sample(
            self._AllNodeNames,
            math.floor(len(self._AllNodeNames) * float(percent_voltage) / 100),
        )
        with open(os.path.join("sensors", "voltage_ids.json"), "w") as fp:
            json.dump(voltage_subset, fp, indent=4)

        random.seed(real_seed)
        real_subset = random.sample(
            self._AllNodeNames,
            math.floor(len(self._AllNodeNames) * float(percent_real) / 100),
        )
        with open(os.path.join("sensors", "real_ids.json"), "w") as fp:
            json.dump(real_subset, fp, indent=4)

        random.seed(reactive_seed)
        reactive_subset = random.sample(
            self._AllNodeNames,
            math.floor(len(self._AllNodeNames) * float(percent_voltage) / 100),
        )
        with open(os.path.join("sensors", "reactive_ids.json"), "w") as fp:
            json.dump(reactive_subset, fp, indent=4)

    def get_circuit_name(self):
        return self._circuit.Name()

    def get_source_indices(self):
        return self._source_indexes

    def get_node_names(self):
        return self._AllNodeNames

    def load_feeder(self):
        dss.Basic.LegacyModels(True)
        dss.run_command("clear")
        result = dss.run_command("redirect " + self._feeder_file)
        if not result == "":
            raise ValueError("Feeder not loaded: " + result)
        self._circuit = dss.Circuit
        self._AllNodeNames = self._circuit.YNodeOrder()
        self._node_number = len(self._AllNodeNames)
        self._nodes_index = [self._AllNodeNames.index(ii) for ii in self._AllNodeNames]
        self._name_index_dict = {
            ii: self._AllNodeNames.index(ii) for ii in self._AllNodeNames
        }

        self._source_indexes = []
        for Source in dss.Vsources.AllNames():
            self._circuit.SetActiveElement("Vsource." + Source)
            Bus = dss.CktElement.BusNames()[0].upper()
            for phase in range(1, dss.CktElement.NumPhases() + 1):
                self._source_indexes.append(
                    self._AllNodeNames.index(Bus.upper() + "." + str(phase))
                )

        self.setup_vbase()
        self._state = OpenDSSState.LOADED

    def disabled_run(self):
        assert self._state != OpenDSSState.UNLOADED, f"{self._state}"
        dss.run_command("batchedit transformer..* wdg=2 tap=1")
        dss.run_command("batchedit regcontrol..* enabled=false")
        dss.run_command("batchedit vsource..* enabled=false")
        dss.run_command("batchedit isource..* enabled=false")
        dss.run_command("batchedit load..* enabled=false")
        dss.run_command("batchedit generator..* enabled=false")
        dss.run_command("batchedit pvsystem..* enabled=false")
        dss.run_command("Batchedit Capacitor..* enabled=false")
        dss.run_command("batchedit storage..* enabled=false")
        dss.run_command("CalcVoltageBases")
        dss.run_command("set maxiterations=20")
        # solve
        dss.run_command("solve")
        # dss.run_command("export y triplet base_ysparse.csv")
        # dss.run_command("export ynodelist base_nodelist.csv")
        # dss.run_command("export summary base_summary.csv")
        self._state = OpenDSSState.DISABLED_RUN

    def get_y_matrix(self):
        self.disabled_run()
        self._state = OpenDSSState.DISABLED_RUN

        Ysparse = csc_matrix(dss.YMatrix.getYsparse())
        Ymatrix = Ysparse.tocoo()
        new_order = self._circuit.YNodeOrder()
        permute = np.array(permutation(new_order, self._AllNodeNames))
        # inv_permute = np.array(permutation(self._AllNodeNames, new_order))
        return coo_matrix(
            (Ymatrix.data, (permute[Ymatrix.row], permute[Ymatrix.col])),
            shape=Ymatrix.shape,
        )

    def setup_vbase(self):
        self._Vbase_allnode = np.zeros((self._node_number), dtype=np.complex_)
        self._Vbase_allnode_dict = {}
        for ii, node in enumerate(self._AllNodeNames):
            self._circuit.SetActiveBus(node)
            self._Vbase_allnode[ii] = dss.Bus.kVBase() * 1000
            self._Vbase_allnode_dict[node] = self._Vbase_allnode[ii]

    def _ready_to_load_power(self, static):
        if static:
            assert self._state != OpenDSSState.UNLOADED, f"{self._state}"
        else:
            assert self._state == OpenDSSState.SOLVE_AT_TIME, f"{self._state}"

    def get_PQs_load(self, static=False):
        self._ready_to_load_power(static)

        num_nodes = len(self._name_index_dict.keys())

        PQ_names = self._AllNodeNames
        PQ_load = np.zeros((num_nodes), dtype=np.complex_)
        for ld in get_loads(dss, self._circuit):
            self._circuit.SetActiveElement("Load." + ld["name"])
            for ii in range(len(ld["phases"])):
                name = ld["bus1"].upper() + "." + ld["phases"][ii]
                index = self._name_index_dict[name]
                if static:
                    power = complex(ld["kW"], ld["kVar"])
                    PQ_load[index] += power / len(ld["phases"])
                else:
                    power = dss.CktElement.Powers()
                    PQ_load[index] += complex(power[2 * ii], power[2 * ii + 1])
                assert PQ_names[index] == name

        return xr.DataArray(PQ_load, {"bus": PQ_names})

    def get_PQs_pv(self, static=False):
        self._ready_to_load_power(static)

        num_nodes = len(self._name_index_dict.keys())

        PQ_names = self._AllNodeNames
        PQ_PV = np.zeros((num_nodes), dtype=np.complex_)
        for PV in get_pvSystems(dss):
            bus = PV["bus"].split(".")
            if len(bus) == 1:
                bus = bus + ["1", "2", "3"]
            self._circuit.SetActiveElement("PVSystem." + PV["name"])
            for ii in range(len(bus) - 1):
                name = bus[0].upper() + "." + bus[ii + 1]
                index = self._name_index_dict[name]
                if static:
                    power = complex(
                        -1 * PV["kW"], -1 * PV["kVar"]
                    )  # -1 because injecting
                    PQ_PV[index] += power / (len(bus) - 1)
                else:
                    power = dss.CktElement.Powers()
                    PQ_PV[index] += complex(power[2 * ii], power[2 * ii + 1])
                assert PQ_names[index] == name
        return xr.DataArray(PQ_PV, {"bus": PQ_names})

    def get_PQs_gen(self, static=False):
        self._ready_to_load_power(static)

        num_nodes = len(self._name_index_dict.keys())

        PQ_names = self._AllNodeNames
        PQ_gen = np.zeros((num_nodes), dtype=np.complex_)
        for PV in get_Generator(dss):
            bus = PV["bus"]
            self._circuit.SetActiveElement("Generator." + PV["name"])
            for ii in range(len(bus) - 1):
                name = bus[0].upper() + "." + bus[ii + 1]
                index = self._name_index_dict[name]
                if static:
                    power = complex(
                        -1 * PV["kW"], -1 * PV["kVar"]
                    )  # -1 because injecting
                    PQ_gen[index] += power / (len(bus) - 1)
                else:
                    power = dss.CktElement.Powers()
                    PQ_gen[index] += complex(power[2 * ii], power[2 * ii + 1])
                assert PQ_names[index] == name
        return xr.DataArray(PQ_gen, {"bus": PQ_names})

    def get_PQs_cap(self, static=False):
        self._ready_to_load_power(static)

        num_nodes = len(self._name_index_dict.keys())

        PQ_names = self._AllNodeNames
        PQ_cap = np.zeros((num_nodes), dtype=np.complex_)
        for cap in get_capacitors(dss):
            for ii in range(cap["numPhases"]):
                name = cap["busname"].upper() + "." + cap["busphase"][ii]
                index = self._name_index_dict[name]
                if static:
                    power = complex(
                        0, -1 * cap["kVar"]
                    )  # -1 because it's injected into the grid
                    PQ_cap[index] += power / cap["numPhases"]
                else:
                    PQ_cap[index] = complex(0, cap["power"][2 * ii + 1])
                assert PQ_names[index] == name

        return xr.DataArray(PQ_cap, {"bus": PQ_names})

    def get_base_voltages(self):
        return xr.DataArray(self._Vbase_allnode, {"bus": self._AllNodeNames})

    def get_disabled_solve_voltages(self):
        assert self._state == OpenDSSState.DISABLED_SOLVE, f"{self._state}"
        return self._get_voltages()

    def get_voltages_snapshot(self):
        assert self._state == OpenDSSState.SNAPSHOT_RUN, f"{self._state}"
        return self._get_voltages()

    def get_voltages_actual(self):
        """

        :return voltages in actual values:
        """
        assert self._state == OpenDSSState.SOLVE_AT_TIME, f"{self._state}"
        return self._get_voltages()

    def _get_voltages(self):
        assert (
            self._state != OpenDSSState.DISABLED_RUN
            and self._state != OpenDSSState.UNLOADED
        ), f"{self._state}"
        _, name_voltage_dict = get_voltages(self._circuit)
        res_feeder_voltages = np.zeros((len(self._AllNodeNames)), dtype=np.complex_)
        for voltage_name in name_voltage_dict.keys():
            res_feeder_voltages[
                self._name_index_dict[voltage_name]
            ] = name_voltage_dict[voltage_name]

        return xr.DataArray(
            res_feeder_voltages, {"bus": list(name_voltage_dict.keys())}
        )

    def change_obj(self, change_commands: CommandList):
        """set/get an object property.

        Input: objData should be a list of lists of the format,

        objName,objProp,objVal,flg],...]
        objName -- name of the object.
        objProp -- name of the property.
        objVal -- val of the property. If flg is set as 'get', then objVal is not used.
        flg -- Can be 'set' or 'get'

        P.S. In the case of 'get' use a value of 'None' for objVal. The same object i.e.
        objData that was passed in as input will have the result i.e. objVal will be
        updated from 'None' to the actual value.
        Sample call: self._changeObj([['PVsystem.pv1','kVAr',25,'set']])
        self._changeObj([['PVsystem.pv1','kVAr','None','get']])
        """
        assert self._state != OpenDSSState.UNLOADED, f"{self._state}"
        for entry in change_commands.__root__:
            dss.Circuit.SetActiveElement(
                entry.obj_name
            )  # make the required element as active element
            dss.CktElement.Properties(entry.obj_property).Val = entry.val

    def initial_disabled_solve(self):
        assert self._state == OpenDSSState.DISABLED_RUN, f"{self._state}"
        hour = 0
        second = 0
        dss.run_command(
            f"set mode=yearly loadmult=1 number=1 hour={hour} sec={second} stepsize={self._simulation_time_step} "
        )
        dss.run_command("solve")
        self._state = OpenDSSState.DISABLED_SOLVE

    def solve(self, hour, second):
        # This only works if you are not unloaded and not disabled
        assert (
            self._state != OpenDSSState.UNLOADED
            and self._state != OpenDSSState.DISABLED_RUN
        ), f"{self._state}"

        dss.run_command(
            f"set mode=yearly loadmult=1 number=1 hour={hour} sec={second} stepsize={self._simulation_time_step} "
        )
        dss.run_command("solve")
        self._state = OpenDSSState.SOLVE_AT_TIME
