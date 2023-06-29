from oedisi.types.common import BrokerConfig
from fastapi import FastAPI, BackgroundTasks, UploadFile
from fastapi.exceptions import HTTPException
from fastapi.responses import FileResponse
from sender_cosim import run_simulator
import zipfile
import uvicorn
import socket
import time
import json
import sys
import os

app = FastAPI()

base_path = os.getcwd()


@app.get("/")
def read_root():
    hostname = socket.gethostname()
    host_ip = socket.gethostbyname(hostname)
    return {"hostname": hostname, "host ip": host_ip}

@app.get("/sensor/")
async def sensor():
    print(os.getcwd())
    sensor_path = os.path.join(base_path, 'sensors','sensors.json')
    while not os.path.exists(sensor_path):
        time.sleep(0.1)
        print(f"waiting {sensor_path}")
    print("success")
    data = json.load(open(sensor_path, "r"))
    return data

@app.post("/profiles/")
async def upload_profiles(file:UploadFile):
    try:
        data = file.file.read()
        if not file.filename.endswith(".zip"):
            HTTPException(400, "Invalid file type. Only zipped profiles are accepted.")
        
        profile_path = "./profiles"
        
        with open(file.filename, "wb") as f:
            f.write(data)
        
        with zipfile.ZipFile(file.filename, 'r') as zip_ref:
            zip_ref.extractall(profile_path)
        
        if os.path.exists(os.path.join(profile_path, "load_profiles")) and os.path.exists(os.path.join(profile_path, "pv_profiles")):
            return {"reply": "success", "error": False, "action": f"File uploaded to server: {file.filename}" }  
        else:
            HTTPException(400, "Invalid user defined profile structure. See OEDISI documentation.")

    except Exception as e:
        HTTPException(500, "Unknown error while uploading userdefined opendss profiles.")

@app.post("/model/")
async def upload_model(file:UploadFile):
    try:
        data = file.file.read()
        if not file.filename.endswith(".zip"):
            HTTPException(400, "Invalid file type. Only zipped opendss models are accepted.")
        
        model_path = "./opendss"
        
        with open(file.filename, "wb") as f:
            f.write(data)
        
        with zipfile.ZipFile(file.filename, 'r') as zip_ref:
            zip_ref.extractall(model_path)
        
        if os.path.exists(os.path.join(model_path, "master.dss")):
            return {"reply": "success", "error": False, "action": f"File uploaded to server: {file.filename}" }    
        else:
            HTTPException(400, "A valid opendss model should have a master.dss file.")
    except Exception as e:
        HTTPException(500, "Unknown error while uploading userdefined opendss model.")

@app.post("/run/")
async def run_feeder(broker_config:BrokerConfig, background_tasks: BackgroundTasks):
    try:
        background_tasks.add_task(run_simulator, broker_config)
        return {"reply": "success", "error": False}
    except Exception as e:
        return {"reply": str(e), "error": True}

if __name__ == "__main__":
    port = int(sys.argv[2])
    uvicorn.run(app, host="0.0.0.0", port=port)