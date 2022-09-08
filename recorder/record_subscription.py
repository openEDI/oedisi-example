import helics as h
import numpy as np
from pydantic import BaseModel
from typing import List
import json
import csv
import pyarrow as pa
from datetime import datetime
from gadal.gadal_types.data_types import MeasurementArray

class Recorder:
    def __init__(self, name, filename, input_mapping):
        self.rng = np.random.default_rng(12345)
        deltat = 0.01
        # deltat = 60.

        # Create Federate Info object that describes the federate properties #
        fedinfo = h.helicsCreateFederateInfo()
        fedinfo.core_name = name
        fedinfo.core_type = h.HELICS_CORE_TYPE_ZMQ
        fedinfo.core_init = "--federates=1"
        print(name)

        h.helicsFederateInfoSetTimeProperty(
            fedinfo, h.helics_property_time_delta, deltat
        )

        self.vfed = h.helicsCreateValueFederate(name, fedinfo)
        print("Value federate created")

        # Register the publication #
        self.sub = self.vfed.register_subscription(
            input_mapping["subscription"], ""
        )
        self.filename = filename

    def run(self):
        # Enter execution mode #
        self.vfed.enter_initializing_mode()
        self.vfed.enter_executing_mode()
        print("Entering execution mode")

        start = True
        granted_time = h.helicsFederateRequestTime(self.vfed, h.HELICS_TIME_MAXTIME)

        with pa.OSFile(self.filename, 'wb') as sink:
            writer = None
            while granted_time < h.HELICS_TIME_MAXTIME:
                print('start',datetime.now())
                print(granted_time)
                # Check that the data is a MeasurementArray type
                json_data = self.sub.json
                json_data['time'] = granted_time
                measurement = MeasurementArray(**self.sub.json)

                measurement_dict = {key: value for key, value in zip(measurement.ids,measurement.values)}
                measurement_dict['time'] = measurement.time.strftime("YY-%m-%d %H:%M:%S")
                print(measurement.time)

                if start:
                    schema_elements = [(key, pa.float64()) for key in measurement.ids]
                    schema_elements.append(('time',pa.string()))
                    schema = pa.schema(schema_elements)
                    writer = pa.ipc.new_file(sink, schema)
                    start = False
                cnt = 0

                writer.write_batch(pa.RecordBatch.from_pylist([
                    measurement_dict
                ]))

                granted_time = h.helicsFederateRequestTime(self.vfed, h.HELICS_TIME_MAXTIME)
                print('end',datetime.now())

            if writer is not None:
                writer.close()
        self.destroy()

    def destroy(self):
        h.helicsFederateDisconnect(self.vfed)
        print("Federate disconnected")
        h.helicsFederateFree(self.vfed)
        h.helicsCloseLibrary()


if __name__ == "__main__":
    with open("static_inputs.json") as f:
        config = json.load(f)
        name = config["name"]
        path = config["filename"]

    with open("input_mapping.json") as f:
        input_mapping = json.load(f)

    sfed = Recorder(name, path, input_mapping)
    sfed.run()
