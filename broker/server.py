from re import L
from fastapi import FastAPI, BackgroundTasks
import helics as h
import requests
import uvicorn
import socket
import yaml
import sys

app = FastAPI()

@app.get("/")
def read_root():
    hostname = socket.gethostname()
    host_ip = socket.gethostbyname(hostname)
    return {"hostname": hostname, "host ip": host_ip}


@app.post("/run/")
async def run_feeder(background_tasks: BackgroundTasks):
    component_map = {}
    with open("docker-compose.yml", "r") as stream:
        config = yaml.safe_load(stream)
    services = config['services']
        
    broker = services.pop('broker')
    broker_ip = broker['networks']['custom-network']['ipv4_address']
    print(broker['ports'][0].split(":"))
    broker_port = int(broker['ports'][0].split(":")[0])

    for service in services:
        ip = services[service]['networks']['custom-network']['ipv4_address']
        port = int(services[service]['ports'][0].split(":")[0])
        component_map[ip] = port
    
    initstring = f"-f {len(component_map)} --name=mainbroker --loglevel=trace --local_interface={broker_ip} --localport={23404}"
    print(f"Broker initaialization string: {initstring}")
    broker = h.helicsCreateBroker("zmq", "", initstring)
    print(broker)
    isconnected = h.helicsBrokerIsConnected(broker)
    print(f"Broker connected: ", isconnected)
    print(component_map)
    for service_ip, service_port in component_map.items():
        url = f'http://{service_ip}:{service_port}/run/'
        print(url)
        myobj = {
            "broker_port" : 23404,
            "broker_ip" : broker_ip,
        }
        print(myobj)
        try:
            reply = requests.post(url, json = myobj)
        except Exception as e:
            reply = None
            print(str(e))
            
        if reply and reply.ok:
            print(f"sucess calling {url}")
            print(f"{reply.text}")
        else:
            print(f"failure calling {url}")
    return

if __name__ == "__main__":
    port = int(sys.argv[2])
    uvicorn.run(app, host="0.0.0.0", port=port)