from fastapi import FastAPI, BackgroundTasks, UploadFile, Request
from fastapi.exceptions import HTTPException
from fastapi.responses import JSONResponse
from sender_cosim import run_simulator
import traceback
import asyncio
import logging
import zipfile
import uvicorn
import socket
import json
import time
import sys
import os

from oedisi.componentframework.system_configuration import ComponentStruct
from oedisi.types.common import ServerReply, HeathCheck, DefaultFileNames
from oedisi.types.common import BrokerConfig

REQUEST_TIMEOUT_SEC = 1200

app = FastAPI()

base_path = os.getcwd()


@app.middleware("http")
async def timeout_middleware(request: Request, call_next):
    try:
        return await asyncio.wait_for(call_next(request), timeout=REQUEST_TIMEOUT_SEC)
    except asyncio.TimeoutError:
        endpoint = str(request.url).replace(str(request.base_url), "").replace("/", "")
        if endpoint == "sensor":
            response = ServerReply(
                detail="Request processing time exceeded limit. Upload a model and associated profiles before simulation before starting the simulation."
            ).dict()
            return JSONResponse(response, 504)
        else:
            response = ServerReply(
                detail="Request processing time exceeded limit"
            ).dict()
            return JSONResponse(response, 504)


@app.get("/")
def read_root():
    hostname = socket.gethostname()
    host_ip = socket.gethostbyname(hostname)
    response = HeathCheck(hostname=hostname, host_ip=host_ip).dict()

    return JSONResponse(response, 200)


@app.get("/sensor")
async def sensor():
    logging.info(os.getcwd())
    sensor_path = os.path.join(base_path, "sensors", "sensors.json")
    while not os.path.exists(sensor_path):
        time.sleep(1)
        logging.info(f"waiting {sensor_path}")
    logging.info("success")
    data = json.load(open(sensor_path, "r"))
    return data


@app.post("/profiles")
async def upload_profiles(file: UploadFile):
    try:
        data = file.file.read()
        if not file.filename.endswith(".zip"):
            HTTPException(400, "Invalid file type. Only zipped profiles are accepted.")

        profile_path = "./profiles"

        with open(file.filename, "wb") as f:
            f.write(data)

        with zipfile.ZipFile(file.filename, "r") as zip_ref:
            zip_ref.extractall(profile_path)

        if os.path.exists(
            os.path.join(profile_path, "load_profiles")
        ) and os.path.exists(os.path.join(profile_path, "pv_profiles")):
            response = ServerReply(
                detail=f"File uploaded to server: {file.filename}"
            ).dict()
            return JSONResponse(response, 200)
        else:
            HTTPException(
                400, "Invalid user defined profile structure. See OEDISI documentation."
            )

    except Exception as e:
        HTTPException(
            500, "Unknown error while uploading userdefined opendss profiles."
        )


@app.post("/model")
async def upload_model(file: UploadFile):
    try:
        data = file.file.read()
        if not file.filename.endswith(".zip"):
            HTTPException(
                400, "Invalid file type. Only zipped opendss models are accepted."
            )

        model_path = "./opendss"

        with open(file.filename, "wb") as f:
            f.write(data)

        with zipfile.ZipFile(file.filename, "r") as zip_ref:
            zip_ref.extractall(model_path)

        if os.path.exists(os.path.join(model_path, "master.dss")):
            response = ServerReply(
                detail=f"File uploaded to server: {file.filename}"
            ).dict()
            return JSONResponse(response, 200)

        else:
            HTTPException(400, "A valid opendss model should have a master.dss file.")
    except Exception as e:
        HTTPException(500, "Unknown error while uploading userdefined opendss model.")


@app.post("/run")
async def run_feeder(
    broker_config: BrokerConfig, background_tasks: BackgroundTasks
):  # :BrokerConfig
    logging.info(broker_config)
    try:
        background_tasks.add_task(run_simulator, broker_config)
        response = ServerReply(detail="Task sucessfully added.").dict()

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
