#!/usr/bin/env python3
# Load component framework
from componentframework.basic_component import component_from_json
from componentframework.system_configuration import (
    generate_runner_config,
    WiringDiagram,
)

def bad_type_checker(x):
    "Doesn't do any type checking on the exchange types"
    return True

# We make classes for each component using a type checker
BasicFeeder = component_from_json(
    "BasicFeeder/component_definition.json", bad_type_checker
)
MeasurementComponent = component_from_json(
    "measuring_federate/component_definition.json", bad_type_checker
)
StateEstimatorComponent = component_from_json(
    "wls_federate/component_definition.json", bad_type_checker
)

# Dictionary used to interprety test_system.json:w
component_types = {
    "BasicFeeeder": BasicFeeder,
    "MeasurementComponent": MeasurementComponent,
    "StateEstimatorComponent": StateEstimatorComponent,
}

# Read wiring diagram (lists components, links, and parameters)
wiring_diagram = WiringDiagram.parse_file("test_system.json")
# Generate runner config using wiring diagram and component types
runner_config = generate_runner_config(wiring_diagram, component_types)
with open("test_system_runner.json", "w") as f:
    f.write(runner_config.json(indent=2))
