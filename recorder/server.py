import logging
import math
import os
import socket
import sys
import traceback

import uvicorn
from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from oedisi.types.common import BrokerConfig, HeathCheck, ServerReply
from record_subscription import run_simulator

app = FastAPI()


@app.get("/")
def read_root():
    hostname = socket.gethostname()
    host_ip = socket.gethostbyname(hostname)

    response = HeathCheck(hostname=hostname, host_ip=host_ip).dict()

    return JSONResponse(response, 200)


def find_filenames(path_to_dir=os.getcwd(), suffix=".feather"):
    filenames = os.listdir(path_to_dir)
    return [filename for filename in filenames if filename.endswith(suffix)]


@app.get("/download/")
def download_results():
    file_list = find_filenames()
    if file_list:
        return FileResponse(
            path=file_list[0], filename=file_list[0], media_type="text/mp4"
        )
    else:
        raise HTTPException(status_code=404, detail="No feather file found")


@app.post("/run/")
async def run_model(broker_config: BrokerConfig, background_tasks: BackgroundTasks):
    logger.info(broker_config)
    try:
        background_tasks.add_task(run_simulator, broker_config)
        response = ServerReply(detail=f"Task sucessfully added.").dict()
        return JSONResponse(response, 200)
    except Exception as e:
        err = traceback.format_exc()
        HTTPException(500, str(err))


if __name__ == "__main__":
    port = int(sys.argv[2])
    uvicorn.run(app, host="0.0.0.0", port=port)
