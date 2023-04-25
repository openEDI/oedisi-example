"""Core class to abstract OpenDSS into Feeder class."""
import json
import logging
import math
import os
import random
import time
from enum import Enum
from time import strptime
from typing import Any, Dict, List

import boto3
import numpy as np
import opendssdirect as dss
import xarray as xr
from botocore import UNSIGNED
from botocore.config import Config
from pydantic import BaseModel
from scipy.sparse import coo_matrix, csc_matrix

from dss_functions import (get_capacitors, get_generators, get_loads,
                           get_pvsystems, get_voltages)

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)


def permutation(from_list, to_list):
    """Create permutation representing change in from_list to to_list.

    Specifically, if `permute = permutation(from_list, to_list)`,
    then `permute[i] = j` means that `from_list[i] = to_list[j]`.

    This also means that `to_list[permute] == from_list`, so you
    can convert from indices under to_list to indices under from_list.

    You may view the permutation as a function from `from_list` to `to_list`.
    """
    index_map = {v: i for i, v in enumerate(to_list)}
    return [index_map[v] for v in from_list]



    
    

class FeederConfig(BaseModel):
    """JSON configuration. Special cases S3 sources right now."""

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
    use_sparse_admittance: bool = False


class Command(BaseModel):
    """JSON Configuration for external object commands.

    obj_name -- name of the object.
    obj_prop -- name of the property.
    obj_val -- val of the property.
    """

    obj_name: str
    obj_property: str
    val: Any


class CommandList(BaseModel):
    """List[Command] with JSON parsing."""

    __root__: List[Command]


class XYCurve(BaseModel):
    voltage: List[float] # p.u. in V
    reactive_response: List[float] # p.u. in VArs


class VVCData(BaseModel):
    """JSON configuration for configuring a volt-var curve."""
    xy_curves: List[XYCurve]
    pv_systems: List[List[str]]


class OpenDSSState(Enum):
    """Enum of all OpenDSSStates traversed in a simulation."""

    UNLOADED = 1
    LOADED = 2
    SNAPSHOT_RUN = 3
    SOLVE_AT_TIME = 4
    DISABLED_RUN = 5
    DISABLED_SOLVE = 6
    DISABLED = 7


    
class FeederSimulator(object):
    """A simple class that handles publishing the solar forecast."""

    # Private variables initialized later
    _feeder_file: str
    _AllNodeNames: List[str]
    _source_indexes: List[int]
    _nodes_index: List[int]
    _name_index_dict: Dict[str, int]

    def __init__(self, config: FeederConfig):
        """Create a ``FeederSimulator`` object."""
        self._state = OpenDSSState.UNLOADED
        self._opendss_location = config.opendss_location
        self._profile_location = config.profile_location
        self._sensor_location = config.sensor_location
        self._use_smartds = config.use_smartds

        self._start_time = int(
            time.mktime(strptime(config.start_date, "%Y-%m-%d %H:%M:%S"))
        )
        self._run_freq_sec = config.run_freq_sec
        self._simulation_step = config.start_time_index
        self._number_of_timesteps = config.number_of_timesteps
        self._vmult = 0.001

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
        """Run snapshot of simuation without specifying a time.

        Used for initialization.
        """
        assert self._state != OpenDSSState.UNLOADED, f"{self._state}"
        dss.Text.Command("Batchedit Load..* enabled=yes")
        dss.Text.Command("Batchedit Vsource..* enabled=yes")
        dss.Text.Command("Batchedit Isource..* enabled=yes")
        dss.Text.Command("Batchedit Generator..* enabled=yes")
        dss.Text.Command("Batchedit PVsystem..* enabled=yes")
        dss.Text.Command("Batchedit Capacitor..* enabled=yes")
        dss.Text.Command("Batchedit Storage..* enabled=no")
        dss.Text.Command("CalcVoltageBases")
        dss.Text.Command("solve mode=snapshot")
        self._state = OpenDSSState.SNAPSHOT_RUN

    def download_data(self, bucket_name, update_loadshape_location=False):
        """Download data from bucket path."""
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
        voltage_seed=1,
        real_seed=2,
        reactive_seed=3,
    ):
        """Initialize list of sensor locations for the measurement federate."""
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
        """Get name of current opendss circuit."""
        return self._circuit.Name()

    def get_source_indices(self):
        """Get indcies of slack buses."""
        return self._source_indexes

    def get_node_names(self):
        """Get node names in order."""
        return self._AllNodeNames

    def load_feeder(self):
        """Load feeder once downloaded. Relies on legacy mode."""
        # Real solution is kvarlimit with kvarmax
        dss.Basic.LegacyModels(True)
        dss.Text.Command("clear")
        dss.Text.Command("redirect " + self._feeder_file)
        result = dss.Text.Result()
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

    def disable_elements(self):
        """Disable most elements. Used in disabled_run"""
        assert self._state != OpenDSSState.UNLOADED, f"{self._state}"
        #dss.Text.Command("batchedit transformer..* wdg=2 tap=1")
        dss.Text.Command("batchedit regcontrol..* enabled=false")
        dss.Text.Command("batchedit vsource..* enabled=false")
        dss.Text.Command("batchedit isource..* enabled=false")
        dss.Text.Command("batchedit load..* enabled=false")
        dss.Text.Command("batchedit generator..* enabled=false")
        dss.Text.Command("batchedit pvsystem..* enabled=false")
        dss.Text.Command("Batchedit Capacitor..* enabled=false")
        dss.Text.Command("batchedit storage..* enabled=false")
        self._state = OpenDSSState.DISABLED

    def disabled_run(self):
        """Disable most elements and solve. Used for most Y-matrix needs."""
        self.disable_elements()
        assert self._state == OpenDSSState.DISABLED, f"{self._state}"
        dss.Text.Command("CalcVoltageBases")
        dss.Text.Command("set maxiterations=20")
        # solve
        dss.Text.Command("solve")
        self._state = OpenDSSState.DISABLED_RUN

    def get_y_matrix(self):
        """Calculate Y-matrix as a coo-matrix. Disables some elements."""
        self.disabled_run()
        self._state = OpenDSSState.DISABLED_RUN

        Ysparse = csc_matrix(dss.YMatrix.getYsparse())
        Ymatrix = Ysparse.tocoo()
        new_order = self._circuit.YNodeOrder()
        permute = np.array(permutation(new_order, self._AllNodeNames))
        return coo_matrix(
            (Ymatrix.data, (permute[Ymatrix.row], permute[Ymatrix.col])),
            shape=Ymatrix.shape,
        )

    def get_load_y_matrix(self):
        """Calculate Y-matrix as a coo-matrix. Disables most except load."""
        assert self._state == OpenDSSState.SOLVE_AT_TIME, f"{self._state}"
        self.disable_elements()
        dss.Text.Command("batchedit Load..* enabled=true")
        dss.Text.Command("CalcVoltageBases")
        dss.Text.Command("set maxiterations=20")
        # solve
        dss.Text.Command("solve")

        Ysparse = csc_matrix(dss.YMatrix.getYsparse())
        Ymatrix = Ysparse.tocoo()
        new_order = self._circuit.YNodeOrder()
        permute = np.array(permutation(new_order, self._AllNodeNames))

        dss.Text.Command("batchedit Load..* enabled=false")
        self._state = OpenDSSState.DISABLED_RUN

        return coo_matrix(
            (Ymatrix.data, (permute[Ymatrix.row], permute[Ymatrix.col])),
            shape=Ymatrix.shape,
        )

    def setup_vbase(self):
        """Load base voltages into feeder."""
        self._Vbase_allnode = np.zeros((self._node_number), dtype=np.complex_)
        self._Vbase_allnode_dict = {}
        for ii, node in enumerate(self._AllNodeNames):
            self._circuit.SetActiveBus(node)
            self._Vbase_allnode[ii] = dss.Bus.kVBase() * 1000
            self._Vbase_allnode_dict[node] = self._Vbase_allnode[ii]

    def _ready_to_load_power(self, static):
        """Check if opendss state can actually calculate power."""
        if static:
            assert self._state != OpenDSSState.UNLOADED, f"{self._state}"
        else:
            assert self._state == OpenDSSState.SOLVE_AT_TIME, f"{self._state}"

    def get_PQs_load(self, static=False):
        """Get active and reactive power of loads as xarray."""
        self._ready_to_load_power(static)

        all_node_names = set(self._AllNodeNames)
        PQs: List[complex] = []
        node_names: List[str] = []
        pq_names: List[str] = []
        # PQ_load = np.zeros((num_nodes), dtype=np.complex_)
        for ld in get_loads(dss, self._circuit):
            self._circuit.SetActiveElement("Load." + ld["name"])
            current_pq_name = dss.CktElement.Name()
            for ii in range(len(ld["phases"])):
                node_name = ld["bus1"].upper() + "." + ld["phases"][ii]
                assert (
                    node_name in all_node_names
                ), f"{node_name} for {current_pq_name} not found"
                if static:
                    power = complex(ld["kW"], ld["kVar"])
                    PQs.append(power / len(ld["phases"]))
                else:
                    power = dss.CktElement.Powers()
                    PQs.append(complex(power[2 * ii], power[2 * ii + 1]))
                pq_names.append(current_pq_name)
                node_names.append(node_name)
        pq_xr = xr.DataArray(
            PQs,
            dims=("eqnode",),
            coords={
                "equipment_ids": ("eqnode", pq_names),
                "ids": ("eqnode", node_names),
            },
        )
        return pq_xr.sortby(pq_xr.ids)

    def get_PQs_pv(self, static=False):
        """Get active and reactive power of PVSystems as xarray."""
        self._ready_to_load_power(static)

        all_node_names = set(self._AllNodeNames)
        PQs: List[complex] = []
        node_names: List[str] = []
        pq_names: List[str] = []
        for PV in get_pvsystems(dss):
            bus = PV["bus"].split(".")
            if len(bus) == 1:
                bus = bus + ["1", "2", "3"]
            self._circuit.SetActiveElement("PVSystem." + PV["name"])
            current_pq_name = dss.CktElement.Name()
            for ii in range(len(bus) - 1):
                node_name = bus[0].upper() + "." + bus[ii + 1]
                assert (
                    node_name in all_node_names
                ), f"{node_name} for {current_pq_name} not found"
                if static:
                    power = complex(
                        -1 * PV["kW"], -1 * PV["kVar"]
                    )  # -1 because injecting
                    PQs.append(power / (len(bus) - 1))
                else:
                    power = dss.CktElement.Powers()
                    PQs.append(complex(power[2 * ii], power[2 * ii + 1]))
                pq_names.append(current_pq_name)
                node_names.append(node_name)
        pq_xr = xr.DataArray(
            PQs,
            dims=("eqnode",),
            coords={
                "equipment_ids": ("eqnode", pq_names),
                "ids": ("eqnode", node_names),
            },
        )
        return pq_xr.sortby(pq_xr.ids)

    def get_PQs_gen(self, static=False):
        """Get active and reactive power of Generators as xarray."""
        self._ready_to_load_power(static)

        all_node_names = set(self._AllNodeNames)
        PQs: List[complex] = []
        node_names: List[str] = []
        pq_names: List[str] = []
        for gen in get_generators(dss):
            bus = gen["bus"].split(".")
            if len(bus) == 1:
                bus = bus + ["1", "2", "3"]
            self._circuit.SetActiveElement("Generator." + gen["name"])
            current_pq_name = dss.CktElement.Name()
            for ii in range(len(bus) - 1):
                node_name = bus[0].upper() + "." + bus[ii + 1]
                assert (
                    node_name in all_node_names
                ), f"{node_name} for {current_pq_name} not found"
                if static:
                    power = complex(
                        -1 * gen["kW"], -1 * gen["kVar"]
                    )  # -1 because injecting
                    PQs.append(power / (len(bus) - 1))
                else:
                    power = dss.CktElement.Powers()
                    PQs.append(complex(power[2 * ii], power[2 * ii + 1]))
                pq_names.append(current_pq_name)
                node_names.append(node_name)
        pq_xr = xr.DataArray(
            PQs,
            dims=("eqnode",),
            coords={
                "equipment_ids": ("eqnode", pq_names),
                "ids": ("eqnode", node_names),
            },
        )
        return pq_xr.sortby(pq_xr.ids)

    def get_PQs_cap(self, static=False):
        """Get active and reactive power of Capacitors as xarray."""
        self._ready_to_load_power(static)

        all_node_names = set(self._AllNodeNames)
        PQs: List[complex] = []
        node_names: List[str] = []
        pq_names: List[str] = []
        for cap in get_capacitors(dss):
            current_pq_name = cap["name"]
            for ii in range(cap["numPhases"]):
                node_name = cap["busname"].upper() + "." + cap["busphase"][ii]
                assert (
                    node_name in all_node_names
                ), f"{node_name} for {current_pq_name} not found"
                if static:
                    power = complex(
                        0, -1 * cap["kVar"]
                    )  # -1 because it's injected into the grid
                    PQs.append(power / cap["numPhases"])
                else:
                    PQs.append(complex(0, cap["power"][2 * ii + 1]))
                pq_names.append(current_pq_name)
                node_names.append(node_name)
        pq_xr = xr.DataArray(
            PQs,
            dims=("eqnode",),
            coords={
                "equipment_ids": ("eqnode", pq_names),
                "ids": ("eqnode", node_names),
            },
        )
        return pq_xr.sortby(pq_xr.ids)

    def get_base_voltages(self):
        """Get base voltages xarray. Can be uesd anytime."""
        return xr.DataArray(self._Vbase_allnode, {"ids": self._AllNodeNames})

    def get_disabled_solve_voltages(self):
        """Get voltage xarray when elements are disabled."""
        assert self._state == OpenDSSState.DISABLED_SOLVE, f"{self._state}"
        return self._get_voltages()

    def get_voltages_snapshot(self):
        """Get voltage xarray in snapshot run."""
        assert self._state == OpenDSSState.SNAPSHOT_RUN, f"{self._state}"
        return self._get_voltages()

    def get_voltages_actual(self):
        """Get voltages xarray at current time."""
        assert self._state == OpenDSSState.SOLVE_AT_TIME, f"{self._state}"
        return self._get_voltages()

    def _get_voltages(self):
        assert (
            self._state != OpenDSSState.DISABLED_RUN
            and self._state != OpenDSSState.UNLOADED
        ), f"{self._state}"
        name_voltage_dict = get_voltages(self._circuit)
        res_feeder_voltages = np.zeros((len(self._AllNodeNames)), dtype=np.complex_)
        for voltage_name in name_voltage_dict.keys():
            res_feeder_voltages[
                self._name_index_dict[voltage_name]
            ] = name_voltage_dict[voltage_name]

        return xr.DataArray(
            res_feeder_voltages, {"ids": list(name_voltage_dict.keys())}
        )

    def change_obj(self, change_commands: CommandList):
        """set/get an object property.

        Parameters
        ----------
        change_commands: CommandList (List[Command])


        Examples
        --------
        ``change_obj(CommandList(__root__ = [Command('PVsystem.pv1','kVAr',25)]))``
        """
        assert self._state != OpenDSSState.UNLOADED, f"{self._state}"
        for entry in change_commands.__root__:
            dss.Circuit.SetActiveElement(
                entry.obj_name
            )  # make the required element as active element
            # dss.CktElement.Properties(entry.obj_property).Val = entry.val
            # dss.Properties.Value(entry.obj_property, str(entry.val))
            properties = dss.CktElement.AllPropertyNames()
            element_name = dss.CktElement.Name()
            assert entry.obj_property.lower() in map(
                lambda x: x.lower(), properties
            ), f"{entry.obj_property} not in {properties} for {element_name}"
            dss.Text.Command(f"{entry.obj_name}.{entry.obj_property}={entry.val}")

    def apply_vcc(self, vvc_data: VVCData):
        self.inv_controls = [] # We should loop through and discover
        self.xy_curves = [] # We should loop through and discover
        # Deduplicate xy-curves if necessary.
        # Create new xy-curves
        # Unmap existing inv_controls that use pv_systems
        # Make inv_control for xy curve and assign pv_systems
        return NotImplemented

    def initial_disabled_solve(self):
        """If run is disabled, then we can still solve at 0.0."""
        assert self._state == OpenDSSState.DISABLED_RUN, f"{self._state}"
        hour = 0
        second = 0
        dss.Text.Command(
            f"set mode=yearly loadmult=1 number=1 hour={hour} sec={second} "
            f"stepsize=0"
        )
        dss.Text.Command("solve")
        self._state = OpenDSSState.DISABLED_SOLVE

    def just_solve(self):
        """Solve without setting time or anything. Useful for commands."""
        assert (
            self._state != OpenDSSState.UNLOADED
            and self._state != OpenDSSState.DISABLED_RUN
        ), f"{self._state}"
        dss.Text.Command("solve")

    def solve(self, hour, second):
        """Solve at specified time. Must not be unloaded or disabled."""
        assert (
            self._state != OpenDSSState.UNLOADED
            and self._state != OpenDSSState.DISABLED_RUN
        ), f"{self._state}"

        dss.Text.Command(
            f"set mode=yearly loadmult=1 number=1 hour={hour} sec={second} "
            f"stepsize=0"
        )
        dss.Text.Command("solve")
        self._state = OpenDSSState.SOLVE_AT_TIME

class FeederMapping(BaseModel):
    static_inputs : FeederConfig
    input_mapping : Dict