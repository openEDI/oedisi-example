"""Server for LocalFeeder federate."""

import asyncio
import json
import logging
import os
import socket
import time
import zipfile
from enum import StrEnum

import uvicorn
from ditto.readers.cim_iec_61968_13.reader import Reader as CIMReader
from ditto.readers.cyme.reader import Reader as CymeReader
from ditto.writers.opendss.write import Writer as OpenDSSWriter
from fastapi import BackgroundTasks, FastAPI, Request, UploadFile
from fastapi.exceptions import HTTPException
from fastapi.responses import JSONResponse
from gdm.distribution import DistributionSystem
from loguru import logger
from oedisi.componentframework.system_configuration import ComponentStruct
from oedisi.types.common import BrokerConfig, DefaultFileNames, HeathCheck, ServerReply

from .sender_cosim import run_simulator

REQUEST_TIMEOUT_SEC = 1200

app = FastAPI()

base_path = os.getcwd()


class SupportedFormats(StrEnum):
    """Supported uploaded model formats for the LocalFeeder /model endpoint."""

    CIM = "CIM"
    CYME = "CYME"
    JSON = "JSON"
    OPENDSS = "OPENDSS"


@app.middleware("http")
async def timeout_middleware(request: Request, call_next):
    """Middleware to enforce request timeouts."""
    try:
        return await asyncio.wait_for(call_next(request), timeout=REQUEST_TIMEOUT_SEC)
    except asyncio.TimeoutError:
        endpoint = str(request.url).replace(str(request.base_url), "").replace("/", "")
        if endpoint == "sensor":
            response = ServerReply(
                detail=(
                    "Request processing time exceeded limit. Upload a model and associated profiles "
                    "before simulation before starting the simulation."
                )
            ).model_dump()
            return JSONResponse(response, 504)
        else:
            response = ServerReply(detail="Request processing time exceeded limit").model_dump()
            return JSONResponse(response, 504)


@app.get("/")
def read_root():
    """Health check endpoint."""
    hostname = socket.gethostname()
    host_ip = socket.gethostbyname(hostname)
    response = HeathCheck(hostname=hostname, host_ip=host_ip).model_dump()

    return JSONResponse(response, 200)


@app.get("/sensor")
async def sensor():
    """Endpoint to retrieve sensor data."""
    logging.info(os.getcwd())
    sensor_path = os.path.join(base_path, "sensors", "sensors.json")
    while not os.path.exists(sensor_path):
        time.sleep(1)
        logging.info(f"waiting {sensor_path}")
    logging.info("success")
    data = json.load(open(sensor_path))
    return data


@app.post("/profiles")
async def upload_profiles(file: UploadFile):
    """Endpoint to upload power profiles."""
    try:
        data = file.file.read()
        if not file.filename.endswith(".zip"):
            HTTPException(400, "Invalid file type. Only zipped profiles are accepted.")

        profile_path = "./profiles"

        with open(file.filename, "wb") as f:
            f.write(data)

        with zipfile.ZipFile(file.filename, "r") as zip_ref:
            zip_ref.extractall(profile_path)

        if os.path.exists(os.path.join(profile_path, "load_profiles")) and os.path.exists(
            os.path.join(profile_path, "pv_profiles")
        ):
            response = ServerReply(detail=f"File uploaded to server: {file.filename}").model_dump()
            return JSONResponse(response, 200)
        else:
            HTTPException(400, "Invalid user defined profile structure. See OEDISI documentation.")

    except Exception as err:
        raise HTTPException(500, "Unknown error while uploading userdefined opendss profiles.") from err


def _read_cim(model_path: str) -> DistributionSystem:
    """Read a CIM model."""
    logger.info(f"Reading CIM model from path: {model_path}")
    cim_file = None
    for root, _, files in os.walk(model_path):
        for filename in files:
            if filename.lower().endswith(".xml"):
                cim_file = os.path.join(root, filename)
                break
        if cim_file is not None:
            break

    assert cim_file is not None, f"No XML file found in {model_path} for CIM input"

    reader = CIMReader(cim_file)
    reader.read()
    logger.info(f"Successfully read CIM model from {cim_file}")
    return reader.get_system()


def _read_cyme(model_path: str) -> DistributionSystem:
    """Read a CYME model."""
    logger.info(f"Reading CYME model from path: {model_path}")
    network_file = None
    equipment_file = None
    load_file = None

    for root, _, files in os.walk(model_path):
        for filename in files:
            filename_lower = filename.lower()
            full_path = os.path.join(root, filename)
            if filename_lower == "network.txt" and network_file is None:
                network_file = full_path
            elif filename_lower == "equipment.txt" and equipment_file is None:
                equipment_file = full_path
            elif filename_lower == "load.txt" and load_file is None:
                load_file = full_path

    assert network_file is not None, f"No Network.txt found in {model_path} for CYME input"
    assert equipment_file is not None, f"No Equipment.txt found in {model_path} for CYME input"

    reader = CymeReader(network_file, equipment_file, load_file)
    logger.info(f"Successfully read CYME model from {model_path}")
    return reader.system


def _read_json(model_path: str) -> DistributionSystem:
    """Read a JSON model."""
    logger.info(f"Reading JSON model from path: {model_path}")
    json_file = None
    for root, _, files in os.walk(model_path):
        for filename in files:
            if filename.lower().endswith(".json"):
                json_file = os.path.join(root, filename)
                break
        if json_file is not None:
            break

    assert json_file is not None, f"No JSON file found in {model_path} for JSON input"
    system = DistributionSystem.from_json(json_file)
    logger.info(f"Successfully read JSON model from {json_file}")
    return system


def _convert_to_opendss(system: DistributionSystem, opendss_path: str):
    """Convert a model to OpenDSS format."""
    logger.info(f"Converting model to OpenDSS format at path: {opendss_path}")
    writer = OpenDSSWriter(system)
    writer.write(opendss_path, separate_substations=False, separate_feeders=False)
    logger.info(f"Successfully converted model to OpenDSS format at path: {opendss_path}")


@app.post("/model")
async def upload_model(file: UploadFile, supported_format: SupportedFormats = SupportedFormats.OPENDSS):
    """Endpoint to upload OpenDSS models."""
    try:
        data = file.file.read()
        model_path = "./converted_model"
        opendss_path = "./opendss"
        if not file.filename.endswith(".zip"):
            HTTPException(400, "Invalid file type. Only zipped opendss models are accepted.")
        with open(file.filename, "wb") as f:
            f.write(data)
        with zipfile.ZipFile(file.filename, "r") as zip_ref:
            zip_ref.extractall(model_path)

        system = None
        if supported_format == SupportedFormats.CIM:
            system = _read_cim(model_path)
        elif supported_format == SupportedFormats.CYME:
            system = _read_cyme(model_path)
        elif supported_format == SupportedFormats.JSON:
            system = _read_json(model_path)

        if system:
            _convert_to_opendss(system, opendss_path)
        else:
            data = file.file.read()
            if not file.filename.endswith(".zip"):
                HTTPException(400, "Invalid file type. Only zipped opendss models are accepted.")
            model_path = "./opendss"
            with open(file.filename, "wb") as f:
                f.write(data)
            with zipfile.ZipFile(file.filename, "r") as zip_ref:
                zip_ref.extractall(model_path)

        if os.path.exists(os.path.join(model_path, "master.dss")):
            response = ServerReply(detail=f"File uploaded to server: {file.filename}").model_dump()
            return JSONResponse(response, 200)

        else:
            HTTPException(400, "A valid opendss model should have a master.dss file.")
    except Exception as err:
        raise HTTPException(500, "Unknown error while uploading userdefined opendss model.") from err


@app.post("/run")
async def run_feeder(broker_config: BrokerConfig, background_tasks: BackgroundTasks):  # :BrokerConfig
    """Run the feeder simulator."""
    logging.info(broker_config)
    try:
        background_tasks.add_task(run_simulator, broker_config)
        response = ServerReply(detail="Task sucessfully added.").model_dump()

        return JSONResponse(response, 200)
    except Exception as err:
        raise HTTPException(500, str(err)) from err


@app.post("/configure")
async def configure(component_struct: ComponentStruct):
    """Configure the feeder simulator."""
    component = component_struct.component
    params = component.parameters
    params["name"] = component.name
    links = {}
    for link in component_struct.links:
        links[link.target_port] = f"{link.source}/{link.source_port}"
    json.dump(links, open(DefaultFileNames.INPUT_MAPPING.value, "w"))
    json.dump(params, open(DefaultFileNames.STATIC_INPUTS.value, "w"))
    response = ServerReply(detail="Successfully updated configuration files.").model_dump()
    return JSONResponse(response, 200)


def main():
    """Entry point for localfeeder-server console script."""
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "5678")))


if __name__ == "__main__":
    main()
