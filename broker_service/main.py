from unicodedata import name
from gadal.gadal_types.mapped_federates import AppPort
from gadal.gadal_tools.cli_tools import get_basic_component
from gadal.componentframework.system_configuration import generate_runner_config, WiringDiagram
from fastapi import FastAPI, BackgroundTasks
from broker_config import Broker
import helics as h
import subprocess
import uvicorn
import socket
import time
import os

app = FastAPI()

def build_wiring_diagram(config):
    components = config.components
    system = config.system
    target_directory = os.getcwd()
    component_types = {
            name: get_basic_component(component_file)
            for name, component_file in components.items()
        }

    wiring_diagram = WiringDiagram.parse_file(system)

    runner_config = generate_runner_config(
        wiring_diagram, component_types, target_directory=target_directory
    )
    
    with open(f"{target_directory}/system_runner.json", "w") as f:
        f.write(runner_config.json(indent=2))
    

def create_broker(broker_config:Broker):
    debug_level = "trace" if broker_config.debug else "warning"
    initstring = "-f {} -n {} --localport={} --localinterface={} --loglevel={}".format(
        broker_config.num_federates,
        broker_config.broker_name,
        broker_config.broker_port,
        broker_config.broker_ip,
        debug_level
        
    )
    print(initstring)
    broker = h.helicsCreateBroker("zmq", "", initstring)
    count = 0
    while h.helicsBrokerIsConnected(broker):
        time.sleep(1)
        if count % 60 == 0:
            print("Broker is live")
        count += 1
        
    print("Broker run complete")
    return 

@app.get("/")
def read_root():
    hostname = socket.gethostname()
    host_ip = socket.gethostbyname(hostname)
    return {"hostname": hostname, "host ip": host_ip}
    
@app.post("/run")
async def run_model(broker_config:Broker, background_tasks:BackgroundTasks):
    try:
        background_tasks.add_task(create_broker, broker_config)
        return {"reply": "sucess", "error": False}
    except Exception as e:
        return {"reply": str(e), "error": True}
        
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=AppPort.broker.value)