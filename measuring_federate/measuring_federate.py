import helics as h
import numpy as np
from pydantic import BaseModel
from typing import List
import scipy.io
import json


class MeasurementConfig(BaseModel):
    name: str
    gaussian_variance: float
    measurement_file: str
    random_percent: float


def get_indices(labelled_array, indices):
    "Get list of indices in the topology for each index of the labelled array"
    inv_map = {v: i for i, v in enumerate(indices)}
    return [inv_map[v] for v in labelled_array.unique_ids]


def reindex(labelled_array, indices):
    inv_map = {v: i for i, v in enumerate(labelled_array.unique_ids)}
    for i in inv_map:
        print(i)
    return LabelledArray(array=[
        labelled_array.array[inv_map[i]] for i in indices
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
        self.sub_measurement = self.vfed.register_subscription(
            input_mapping["subscription"], ""
        )

        #TODO: find better way to determine what the name of this federate instance is than looking at the subscription
        publication = input_mapping["subscription"].split('/')[1]
        self.pub_measurement = self.vfed.register_publication(
            publication, h.HELICS_DATA_TYPE_STRING, ""
        )

        self.gaussian_variance = config.gaussian_variance
        self.measurement_file = config.measurement_file
        self.random_percent = config.random_percent

    def transform(self, array: LabelledArray, unique_ids):
        new_array = reindex(array, unique_ids)
        return apply(
            lambda x: x + self.rng.normal(scale=np.sqrt(self.gaussian_variance)),
            new_array
        )

    def run(self):
        # Enter execution mode #
        self.vfed.enter_executing_mode()
        print("Entering execution mode")

        granted_time = h.helicsFederateRequestTime(self.vfed, h.HELICS_TIME_MAXTIME)
        while granted_time < h.HELICS_TIME_MAXTIME:
            print(granted_time)
            json_data = self.sub_measurement.json
            measurement = MeasurementArray(**json_data)

            with open(self.measurement_file,'r') as fp:
                self.measurement = json.load(fp)
            measurement_transformed = self.transform(measurement)
            print("measured transformed")
            print(measurement_transformed)

            self.pub_voltages.publish(measurement_transformed.json())

            granted_time = h.helicsFederateRequestTime(self.vfed, h.HELICS_TIME_MAXTIME)

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
