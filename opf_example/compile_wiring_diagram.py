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
OPFFeeder = component_from_json(
    "dss_federate_123PV/component_definition.json", bad_type_checker
)
OPFComponent = component_from_json(
    "opf_federate_123PV/component_definition.json", bad_type_checker
)
#Recorder = component_from_json(
#    "recorder/component_definition.json", bad_type_checker
#)

# Dictionary used to interpret test_system.json
component_types = {
    "OPFFeeder": OPFFeeder,
    "OPFComponent": OPFComponent,
}

# Read wiring diagram (lists components, links, and parameters)
wiring_diagram = WiringDiagram.parse_file("system.json")
# Generate runner config using wiring diagram and component types
runner_config = generate_runner_config(wiring_diagram, component_types)
with open("system_runner.json", "w") as f:
    f.write(runner_config.json(indent=2))
