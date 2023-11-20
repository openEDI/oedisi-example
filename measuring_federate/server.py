from fastapi import FastAPI, BackgroundTasks, HTTPException
from oedisi.types.common import BrokerConfig
from measuring_federate import run_simulator
from fastapi.responses import JSONResponse
import uvicorn
import socket
import requests
import sys
import json
import traceback

import math
from oedisi.types.common import ServerReply, HeathCheck

app = FastAPI()



@app.get("/")
async def read_root():
    hostname = socket.gethostname()
    host_ip = socket.gethostbyname(hostname)
    response = HeathCheck(
        hostname = hostname,
        host_ip = host_ip
    ).dict()
    return JSONResponse(response, 200)
    
@app.post("/run/")
async def run_model(broker_config:BrokerConfig, background_tasks: BackgroundTasks):
    try:
        print(broker_config)
        feeder_ip = broker_config.services['oedisi_feeder']['networks']['custom-network']['ipv4_address']
        feeder_port = int(broker_config.services['oedisi_feeder']['ports'][0].split(":")[0])
       
        url =f"http://{feeder_ip}:{feeder_port}/sensor/"
        print(url)
        reply = requests.get(url)
        sensor_data = reply.json()
        with open("sensors.json", "w") as outfile:
            json.dump(sensor_data, outfile)

        background_tasks.add_task(run_simulator, broker_config)
        response = ServerReply(
            detail = f"Task sucessfully added."
        ).dict() 
        return JSONResponse(response, 200)
    except Exception as e:
        err = traceback.format_exc()
        HTTPException(500,str(err))

if __name__ == "__main__":
    port = int(sys.argv[2])
    uvicorn.run(app, host="0.0.0.0", port=port)

