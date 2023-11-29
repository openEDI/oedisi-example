from oedisi.types.common import BrokerConfig
from state_estimator_federate import run_simulator
from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.responses import JSONResponse
import traceback
import uvicorn
import socket
import sys

from oedisi.types.common import ServerReply, HeathCheck

app = FastAPI()


@app.get("/")
def read_root():
    hostname = socket.gethostname()
    host_ip = socket.gethostbyname(hostname)
    response = HeathCheck(
        hostname=hostname,
        host_ip=host_ip
    ).dict()
    return JSONResponse(response, 200)


@app.post("/run/")
async def run_model(broker_config: BrokerConfig, background_tasks: BackgroundTasks):
    print(broker_config)
    try:
        background_tasks.add_task(run_simulator, broker_config)
        response = ServerReply(
            detail="Task sucessfully added."
        ).dict()
        return JSONResponse(response, 200)
    except Exception as e:
        err = traceback.format_exc()
        HTTPException(500, str(err))


if __name__ == "__main__":
    port = int(sys.argv[2])
    uvicorn.run(app, host="0.0.0.0", port=port)
