from oedisi.types.common import BrokerConfig
from opf_federate import EchoFederate
from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.responses import JSONResponse
import traceback
import uvicorn
import socket
import sys

from oedisi.types.common import ServerReply

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
    return JSONResponse({"hostname": hostname, "host_ip": host_ip}, 200)


@app.post("/run/")
async def run_model(broker_config: BrokerConfig, background_tasks: BackgroundTasks):
    federate = EchoFederate(broker_config)
    try:
        background_tasks.add_task(federate.run)
        return {"reply": "success", "error": False}
    except Exception as e:
        return {"reply": str(e), "error": True}


if __name__ == "__main__":
    port = int(sys.argv[2])
    uvicorn.run(app, host="0.0.0.0", port=port)
