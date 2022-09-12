#!/usr/bin/env python3
# Load component framework
from gadal.componentframework.basic_component import component_from_json
from gadal.componentframework.system_configuration import (
    generate_runner_config,
    WiringDiagram,
)

def bad_type_checker(type, x):
    "Doesn't do any type checking on the exchange types"
    return True

# We make classes for each component using a type checker
AWSFeeder = component_from_json(
    "AWSFeeder/component_definition.json", bad_type_checker
)
LocalFeeder = component_from_json(
    "LocalFeeder/component_definition.json", bad_type_checker
)
MeasurementComponent = component_from_json(
    "measuring_federate/component_definition.json", bad_type_checker
)
StateEstimatorComponent = component_from_json(
    "wls_federate/component_definition.json", bad_type_checker
)
Recorder = component_from_json(
    "recorder/component_definition.json", bad_type_checker
)
IEEE123Feeder = component_from_json(
    "dss_federate_123PV/component_definition.json", bad_type_checker
)

# Dictionary used to interpret test_system.json
component_types = {
    "LocalFeeder": LocalFeeder,
    "AWSFeeder": AWSFeeder,
    "MeasurementComponent": MeasurementComponent,
    "StateEstimatorComponent": StateEstimatorComponent,
    "Recorder": Recorder
}

# Read wiring diagram (lists components, links, and parameters)
wiring_diagram = WiringDiagram.parse_file("test_system.json")
#wiring_diagram.clean_model()
# Generate runner config using wiring diagram and component types
runner_config = generate_runner_config(wiring_diagram, component_types)
with open("test_system_runner.json", "w") as f:
    f.write(runner_config.json(indent=2))
