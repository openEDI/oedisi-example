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




@app.get("/")
def read_root():
    hostname = socket.gethostname()
    host_ip = socket.gethostbyname(hostname)
    return {"hostname": hostname, "host ip": host_ip}

@app.get("/sensor/")
async def sensor():
    print(os.getcwd())
    sensor_path = os.path.join(os.getcwd(), 'sensors','sensors.json')
    while not os.path.exists(sensor_path):
        time.sleep(0.1)
        print("waiting")
    print("success")
    data = json.load(open(sensor_path, "r"))
    return data

@app.post("/model/")
async def upload_model(file:UploadFile):
    try:
        data = file.file.read()
        if not file.filename.endswith(".zip"):
            HTTPException(400, "Invalid file type. Only zipped opendss models are accepted.")
        
        with open(file.filename, "wb") as f:
            f.write(data)
            
        with zipfile.ZipFile(file.filename, 'r') as zip_ref:
            zip_ref.extractall("./")
        
        return {"reply": "success", "error": False, "action": f"File uploaded to server: {file.filename}" }
    except Exception as e:
        return {"reply": str(e), "error": True}

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