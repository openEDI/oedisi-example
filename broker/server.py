from fastapi import FastAPI, BackgroundTasks, UploadFile
from fastapi.exceptions import HTTPException
import helics as h
import logging
import grequests
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

    broker = services.pop('oedisi_broker')
    broker_ip = broker['networks']['custom-network']['ipv4_address']
    api_port = int(broker['ports'][0].split(":")[0])

    for service in services:
        ip = services[service]['networks']['custom-network']['ipv4_address']
        port = int(services[service]['ports'][0].split(":")[0])
        component_map[ip] = port

    initstring = f"-f {len(component_map)} --name=mainbroker --loglevel=trace --local_interface={broker_ip} --localport={23404}"
    logging.info(f"Broker initaialization string: {initstring}")
    broker = h.helicsCreateBroker("zmq", "", initstring)
    logging.info(broker)
    isconnected = h.helicsBrokerIsConnected(broker)
    logging.info(f"Broker connected: " + str(isconnected))
    logging.info(str(component_map))
    replys = []
    for service_ip, service_port in component_map.items():
        url = f'http://{service_ip}:{service_port}/run/'
        logging.info(url)
        myobj = {
            "broker_port" : 23404,
            "broker_ip" : broker_ip,
            "api_port" : api_port,
            "services" : services,
        }
        logging.info(str(myobj))
        replys.append(grequests.post(url, json = myobj))


        # except Exception as e:
        #     reply = None
        #     logging.error(str(e))

        # if reply and reply.ok:
        #     logging.info(f"sucess calling {url}")
        #     logging.info(f"{reply.text}")
        # else:
        #     logging.error(f"failure calling {url}")
    print(grequests.map(replys))
    return

if __name__ == "__main__":
    port = int(sys.argv[2])
    uvicorn.run(app, host="0.0.0.0", port=port)
