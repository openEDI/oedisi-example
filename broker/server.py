import logging
import os
import shutil
import socket
import sys
import time
import traceback
import zipfile

import grequests
import helics as h
import requests
import uvicorn
import yaml
from fastapi import BackgroundTasks, FastAPI, UploadFile
from fastapi.exceptions import HTTPException
from fastapi.responses import FileResponse, JSONResponse
from oedisi.types.common import HeathCheck, ServerReply
from oedisi.tools.broker_utils import get_time_data

app = FastAPI()


def find_filenames(path_to_dir=os.getcwd(), suffix=".feather"):
    filenames = os.listdir(path_to_dir)
    return [filename for filename in filenames if filename.endswith(suffix)]


def read_settings():
    component_map = {}
    with open("docker-compose.yml", "r") as stream:
        config = yaml.safe_load(stream)
    services = config["services"]
    print(services)
    broker = services.pop("oedisi_broker")
    broker_ip = broker["networks"]["custom-network"]["ipv4_address"]
    api_port = int(broker["ports"][0].split(":")[0])

    for service in services:
        ip = services[service]["networks"]["custom-network"]["ipv4_address"]
        port = int(services[service]["ports"][0].split(":")[0])
        component_map[ip] = port

    return services, component_map, broker_ip, api_port


@app.get("/")
def read_root():
    hostname = socket.gethostname()
    host_ip = socket.gethostbyname(hostname)

    response = HeathCheck(hostname=hostname, host_ip=host_ip).dict()

    return JSONResponse(response, 200)


@app.post("/profiles/")
async def upload_profiles(file: UploadFile):
    try:
        services, _, _, _ = read_settings()
        for service in services:
            if "feeder" in service.lower():
                ip = services[service]["networks"]["custom-network"]["ipv4_address"]
                port = int(services[service]["ports"][0].split(":")[0])
                data = file.file.read()
                if not file.filename.endswith(".zip"):
                    HTTPException(
                        400, "Invalid file type. Only zip files are accepted."
                    )
                with open(file.filename, "wb") as f:
                    f.write(data)
                url = f"http://{ip}:{port}/profiles/"
                files = {"file": open(file.filename, "rb")}
                r = requests.post(url, files=files)
                response = ServerReply(detail=r.text).dict()
                return JSONResponse(response, r.status_code)
        raise HTTPException(status_code=404, detail="Unable to upload profiles")
    except Exception as e:
        err = traceback.format_exc()
        raise HTTPException(status_code=500, detail=str(err))


@app.post("/model/")
async def upload_model(file: UploadFile):
    try:
        services, _, _, _ = read_settings()
        for service in services:
            if "feeder" in service.lower():
                ip = services[service]["networks"]["custom-network"]["ipv4_address"]
                port = int(services[service]["ports"][0].split(":")[0])
                data = file.file.read()
                if not file.filename.endswith(".zip"):
                    HTTPException(
                        400, "Invalid file type. Only zip files are accepted."
                    )
                with open(file.filename, "wb") as f:
                    f.write(data)
                url = f"http://{ip}:{port}/model/"
                files = {"file": open(file.filename, "rb")}
                r = requests.post(url, files=files)
                response = ServerReply(detail=r.text).dict()
                return JSONResponse(response, r.status_code)
        raise HTTPException(status_code=404, detail="Unable to upload model")
    except Exception as e:
        err = traceback.format_exc()
        raise HTTPException(status_code=500, detail=str(err))


@app.get("/results/")
def download_results():
    services, _, _, _ = read_settings()
    for service in services:
        if "recorder" in service.lower():
            ip = services[service]["networks"]["custom-network"]["ipv4_address"]
            port = int(services[service]["ports"][0].split(":")[0])
            url = f"http://{ip}:{port}/download/"
            response = requests.get(url)
            with open(f"{service}.feather", "wb") as out_file:
                shutil.copyfileobj(response.raw, out_file)
                time.sleep(2)

    file_path = "results.zip"
    with zipfile.ZipFile(file_path, "w") as zipMe:
        for feather_file in find_filenames():
            zipMe.write(feather_file, compress_type=zipfile.ZIP_DEFLATED)

    try:
        return FileResponse(path=file_path, filename=file_path, media_type="zip")
    except Exception as e:
        raise HTTPException(status_code=404, detail="Failed download")


@app.get("/terminate/")
def terminate_simulation():
    try:
        h.helicsCloseLibrary()
        return JSONResponse({"detail": "Helics broker sucessfully closed"}, 200)
    except Exception as e:
        raise HTTPException(status_code=404, detail="Failed download ")


def run_simulation():
    services, component_map, broker_ip, api_port = read_settings()
    initstring = f"-f {len(component_map)} --name=mainbroker --loglevel=trace --local_interface={broker_ip} --localport=23404"
    logging.info(f"Broker initaialization string: {initstring}")
    broker = h.helicsCreateBroker("zmq", "", initstring)
    app.state.broker = broker
    logging.info(broker)
    isconnected = h.helicsBrokerIsConnected(broker)
    logging.info(f"Broker connected: {isconnected}")
    logging.info(str(component_map))
    replies = []
    for service_ip, service_port in component_map.items():
        url = f"http://{service_ip}:{service_port}/run/"
        print(url)
        myobj = {
            "broker_port": 23404,
            "broker_ip": broker_ip,
            "api_port": api_port,
            "services": services,
        }
        replies.append(grequests.post(url, json=myobj))
    grequests.map(replies)
    while h.helicsBrokerIsConnected(broker):
        time.sleep(1)
    h.helicsCloseLibrary()

    return


@app.post("/run/")
async def run_feeder(background_tasks: BackgroundTasks):
    try:
        background_tasks.add_task(run_simulation)
        response = ServerReply(detail="Task sucessfully added.").dict()
        return JSONResponse({"detail": response}, 200)
    except Exception as e:
        err = traceback.format_exc()
        raise HTTPException(status_code=404, detail=str(err))


@app.get("/status/")
async def status():
    try:
        name_2_timedata = {}
        connected = h.helicsBrokerIsConnected(app.state.broker)
        if connected:
            for time_data in get_time_data(app.state.broker):
                if (time_data.name not in name_2_timedata) or (
                    name_2_timedata[time_data.name] != time_data
                ):
                    name_2_timedata[time_data.name] = time_data
        return {"connected": connected, "timedata": name_2_timedata, "error": False}
    except AttributeError as e:
        return {"reply": str(e), "error": True}


if __name__ == "__main__":
    port = int(sys.argv[2])
    uvicorn.run(app, host="0.0.0.0", port=port)
