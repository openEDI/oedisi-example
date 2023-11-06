from fastapi import FastAPI, BackgroundTasks, UploadFile
from fastapi.exceptions import HTTPException
from fastapi.responses import FileResponse
import helics as h
import grequests
import traceback
import requests
import uvicorn
import logging
import socket
import shutil
import time
import yaml
import sys
import os

app = FastAPI()

def find_filenames(path_to_dir=os.getcwd(), suffix=".feather" ):
    filenames = os.listdir(path_to_dir)
    return [ filename for filename in filenames if filename.endswith( suffix ) ]
    
def read_settings():
    component_map = {}
    with open("docker-compose.yml", "r") as stream:
        config = yaml.safe_load(stream)
    services = config['services']
    print(services)
    broker = services.pop('oedisi_broker')
    broker_ip = broker['networks']['custom-network']['ipv4_address']
    api_port = int(broker['ports'][0].split(":")[0])

    for service in services:
        ip = services[service]['networks']['custom-network']['ipv4_address']
        port = int(services[service]['ports'][0].split(":")[0])
        component_map[ip] = port
        
    return  services, component_map, broker_ip, api_port

@app.get("/")
def read_root():
    hostname = socket.gethostname()
    host_ip = socket.gethostbyname(hostname)
    return {"hostname": hostname, "host ip": host_ip}

@app.get("/results/")
def download_results():
    services, _, _, _ = read_settings()
    for service in services:
        if "oedisi_recorder" in service.lower():
            ip = services[service]['networks']['custom-network']['ipv4_address']
            port = int(services[service]['ports'][0].split(":")[0])
            url = f'http://{ip}:{port}/download/'
            response = requests.get(url)
            with open(f'{service}.feather', 'wb') as out_file:
                shutil.copyfileobj(response.raw, out_file)
                time.sleep(2)
        
    file_list  = find_filenames()
    if file_list:
        return FileResponse(path=file_list[0], filename=file_list[0], media_type='text/mp4')
    else:
        raise HTTPException(status_code=404, detail="No feather file found")

def run_simulation(services, component_map, broker_ip, api_port):
    
    initstring = f"-f {len(component_map)} --name=mainbroker --loglevel=trace --local_interface={broker_ip} --localport={23404}"
    logging.info(f"Broker initaialization string: {initstring}")
    broker = h.helicsCreateBroker("zmq", "", initstring)
    logging.info(broker)
    isconnected = h.helicsBrokerIsConnected(broker)
    logging.info(f"Broker connected: " + str(isconnected))
    logging.info(str(component_map))
    replies = []
    for service_ip, service_port in component_map.items():
        url = f'http://{service_ip}:{service_port}/run/'
        print(url)
        myobj = {
            "broker_port" : 23404,
            "broker_ip" : broker_ip,
            "api_port" : api_port,
            "services" : services,
        }
        replies.append(grequests.post(url, json = myobj))
        
    print(grequests.map(replies))
    return


@app.post("/run/")
async def run_feeder(background_tasks: BackgroundTasks): 
    data_input = read_settings()
    try:
        background_tasks.add_task(run_simulation, *data_input)
    except Exception as e:
        err = traceback.format_exc()
        raise HTTPException(status_code=404, detail=str(err))

if __name__ == "__main__":
    port = int(sys.argv[2])
    uvicorn.run(app, host="0.0.0.0", port=port)
