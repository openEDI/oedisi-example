from fastapi import FastAPI, BackgroundTasks, HTTPException
from state_estimator_federate import run_simulator
from oedisi.types.common import BrokerConfig
from fastapi.responses import JSONResponse
import traceback
import uvicorn
import socket
import json
import sys
import os

from oedisi.componentframework.system_configuration import ComponentStruct
from oedisi.types.common import ServerReply, HeathCheck, DefaultFileNames

app = FastAPI()


@app.get("/")
def read_root():
    hostname = socket.gethostname()
    host_ip = socket.gethostbyname(hostname)
    response = HeathCheck(
        hostname=hostname,
        host_ip=host_ip
    ).dict()
    return JSONResponse(response, 200)


@app.post("/run")
async def run_model(broker_config: BrokerConfig, background_tasks: BackgroundTasks):
    print(broker_config)
    try:
        background_tasks.add_task(run_simulator, broker_config)
        response = ServerReply(
            detail="Task sucessfully added."
        ).dict()
        return JSONResponse(response, 200)
    except Exception as e:
        err = traceback.format_exc()
        HTTPException(500, str(err))


@app.post("/configure")
async def configure(component_struct:ComponentStruct): 
    component = component_struct.component
    params = component.parameters
    params["name"] = component.name
    links = {}
    for link in component_struct.links:
        links[link.target_port] = f"{link.source}/{link.source_port}"
    json.dump(links , open(DefaultFileNames.INPUT_MAPPING.value, "w"))
    json.dump(params , open(DefaultFileNames.STATIC_INPUTS.value, "w"))
    response = ServerReply(
            detail = f"Sucessfully updated configuration files."
        ).dict() 
    return JSONResponse(response, 200)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ['PORT']))
