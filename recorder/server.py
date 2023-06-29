from fastapi import FastAPI, BackgroundTasks, HTTPException
from record_subscription import run_simulator
from oedisi.types.common import BrokerConfig
from fastapi.responses import FileResponse
import uvicorn
import socket
import sys
import os

app = FastAPI()

@app.get("/")
def read_root():
    hostname = socket.gethostname()
    host_ip = socket.gethostbyname(hostname)
    return {"hostname": hostname, "host ip": host_ip}

def find_filenames(path_to_dir=os.getcwd(), suffix=".feather" ):
    filenames = os.listdir(path_to_dir)
    return [ filename for filename in filenames if filename.endswith( suffix ) ]

@app.get("/download/")
def read_root():
    file_list  = find_filenames()
    if file_list:
        return FileResponse(path=file_list[0], filename=file_list[0], media_type='text/mp4')
    else:
        raise HTTPException(status_code=404, detail="No feather file found")

@app.post("/run/")
async def run_model(broker_config:BrokerConfig, background_tasks: BackgroundTasks):
    try:
        background_tasks.add_task(run_simulator, broker_config)
        return {"reply": "success", "error": False}
    except Exception as e:
        return {"reply": str(e), "error": True}

if __name__ == "__main__":
    port = int(sys.argv[2])
    uvicorn.run(app, host="0.0.0.0", port=port)