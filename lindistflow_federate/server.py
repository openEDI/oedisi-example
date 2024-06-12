from oedisi.types.common import BrokerConfig
from opf_federate import EchoFederate
from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.responses import JSONResponse
import uvicorn
import socket
import traceback
import json
import os

from oedisi.componentframework.system_configuration import ComponentStruct
from oedisi.types.common import ServerReply, HeathCheck, DefaultFileNames

app = FastAPI()


@app.get("/")
def read_root():
    hostname = socket.gethostname()
    host_ip = "127.0.0.1"
    try:
        host_ip = socket.gethostbyname(socket.gethostname())
    except socket.gaierror:
        try:
            host_ip = socket.gethostbyname(socket.gethostname() + ".local")
        except socket.gaierror:
            pass
    response = HeathCheck(hostname=hostname, host_ip=host_ip).dict()
    return JSONResponse(response, 200)


@app.post("/run")
async def run_model(broker_config: BrokerConfig, background_tasks: BackgroundTasks):
    print(broker_config)
    federate = EchoFederate(broker_config)
    try:
        background_tasks.add_task(federate.run)
        response = ServerReply(detail="Task sucessfully added.").dict()
        return JSONResponse(response, 200)
    except Exception as _:
        err = traceback.format_exc()
        HTTPException(500, str(err))


@app.post("/configure")
async def configure(component_struct: ComponentStruct):
    component = component_struct.component
    params = component.parameters
    params["name"] = component.name
    links = {}
    for link in component_struct.links:
        links[link.target_port] = f"{link.source}/{link.source_port}"
    json.dump(links, open(DefaultFileNames.INPUT_MAPPING.value, "w"))
    json.dump(params, open(DefaultFileNames.STATIC_INPUTS.value, "w"))
    response = ServerReply(detail="Sucessfully updated configuration files.").dict()
    return JSONResponse(response, 200)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ["PORT"]))
