import helics as h
import numpy as np
from pydantic import BaseModel
from typing import List
import scipy.io
import json


class Complex(BaseModel):
    real: float
    imag: float


class Topology(BaseModel):
    y_matrix: List[List[Complex]]
    phases: List[float]
    unique_ids: List[str]


class LabelledArray(BaseModel):
    array: List[float]
    unique_ids: List[str]


class PolarLabelledArray(BaseModel):
    magnitudes: List[float]
    angles: List[float]
    unique_ids: List[str]


class MeasurementConfig(BaseModel):
    name: str
    gaussian_variance: float
    voltage_ids: List[str]
    real_power_ids: List[str]
    reactive_power_ids: List[str]


def get_indices(labelled_array, indices):
    "Get list of indices in the topology for each index of the labelled array"
    inv_map = {v: i for i, v in enumerate(indices)}
    return [inv_map[v] for v in labelled_array.unique_ids]


def reindex(labelled_array, indices):
    inv_map = {v: i for i, v in enumerate(labelled_array.unique_ids)}
    return LabelledArray(array=[
        labelled_array[inv_map[i]] for i in indices
    ], unique_ids=indices)


def apply(f, labelled_array):
    return LabelledArray(
        array=list(map(f, labelled_array.array)),
        unique_ids=labelled_array.unique_ids
    )


class MeasurementRelay:
    def __init__(self, config: MeasurementConfig, input_mapping):
        self.rng = np.random.default_rng(12345)
        deltat = 0.01
        # deltat = 60.

        # Create Federate Info object that describes the federate properties #
        fedinfo = h.helicsCreateFederateInfo()
        fedinfo.core_name = config.name
        fedinfo.core_type = h.HELICS_CORE_TYPE_ZMQ
        fedinfo.core_init = "--federates=1"
        print(config.name)

        h.helicsFederateInfoSetTimeProperty(
            fedinfo, h.helics_property_time_delta, deltat
        )

        self.vfed = h.helicsCreateValueFederate(config.name, fedinfo)
        print("Value federate created")

        # Register the publication #
        self.sub_voltage_real = self.vfed.register_subscription(
            input_mapping["voltage_real"], "V"
        )
        self.sub_voltage_imag = self.vfed.register_subscription(
            input_mapping["voltage_imag"], "V"
        )
        self.sub_power_real = self.vfed.register_subscription(
            input_mapping["power_real"], "W"
        )
        self.sub_power_imag = self.vfed.register_subscription(
            input_mapping["power_imag"], "W"
        )
        self.pub_voltages = self.vfed.register_publication(
            "voltages", h.HELICS_DATA_TYPE_STRING, "V"
        )
        self.pub_power_real = self.vfed.register_publication(
            "power_real", h.HELICS_DATA_TYPE_STRING, "W"
        )
        self.pub_power_imag = self.vfed.register_publication(
            "power_imag", h.HELICS_DATA_TYPE_STRING, "W"
        )

        self.gaussian_variances = config.gaussian_variance
        self.voltage_ids = config.voltage_ids
        self.real_power_ids = config.real_power_ids
        self.reactive_power_ids = config.reactive_power_ids

    def transform(self, array: LabelledArray, unique_ids):
        new_array = reindex(array, unique_ids)
        return apply(
            lambda x: x + self.rng.standard_normal(scale=np.sqrt(self.gaussian_variance)),
            new_array
        )

    def run(self):
        # Enter execution mode #
        self.vfed.enter_executing_mode()
        print("Entering execution mode")

        granted_time = h.helicsFederateRequestTime(self.vfed, 100)
        while granted_time < 100:
            print(granted_time)
            voltage_real = LabelledArray.parse_obj(self.sub_voltage_real.json)
            voltage_imag = LabelledArray.parse_obj(self.sub_voltage_imag.json)
            power_real = LabelledArray.parse_obj(self.sub_power_real.json)
            power_imag = LabelledArray.parse_obj(self.sub_power_imag.json)

            assert voltage_real.unique_ids == voltage_imag.unique_ids
            assert voltage_real.unique_ids == power_real.unique_ids
            assert voltage_real.unique_ids == power_imag.unique_ids
            voltage_abs = LabelledArray(
                np.abs(np.array(voltage_real.array) + 1j*np.array(voltage_imag.array)),
                voltage_real.unique_ids
            )
            measured_voltages = self.transform(voltage_abs, self.voltage_ids)
            measured_power_real = self.transform(power_real, self.real_power_ids)
            measured_power_imag = self.transform(power_imag, self.reactive_power_ids)

            self.pub_voltages.publish(measured_voltages.json())
            self.pub_power_real.publish(measured_power_real.json())
            self.pub_power_imag.publish(measured_power_imag.json())

            granted_time = h.helicsFederateRequestTime(self.vfed, 100)

        self.destroy()

    def destroy(self):
        h.helicsFederateDisconnect(self.vfed)
        print("Federate disconnected")
        h.helicsFederateFree(self.vfed)
        h.helicsCloseLibrary()


if __name__ == "__main__":
    with open("static_inputs.json") as f:
        config = MeasurementConfig(**json.load(f))

    with open("input_mapping.json") as f:
        input_mapping = json.load(f)

    sfed = MeasurementRelay(config, input_mapping)
    sfed.run()
