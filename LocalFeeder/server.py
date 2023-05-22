from FeederSimulator import CommandList, FeederConfig, FeederSimulator, FeederMapping
from fastapi import FastAPI, BackgroundTasks
from sender_cosim import go_cosim
import uvicorn
import socket
import sys

app = FastAPI()

def run(model_configs : FeederMapping):
    """Load static_inputs and input_mapping and run JSON."""
    config = model_configs.static_inputs
    input_mapping = model_configs.input_mapping
    sim = FeederSimulator(config)
    go_cosim(sim, config, input_mapping)


@app.get("/")
def read_root():
    hostname = socket.gethostname()
    host_ip = socket.gethostbyname(hostname)
    return {"hostname": hostname, "host ip": host_ip}


@app.post("/run/")
async def run_feeder(feeder_mapping:FeederMapping, background_tasks: BackgroundTasks):
    try:
        background_tasks.add_task(run, feeder_mapping)
        return {"reply": "success", "error": False}
    except Exception as e:
        return {"reply": str(e), "error": True}

if __name__ == "__main__":
    port = int(sys.argv[2])
    uvicorn.run(app, host="0.0.0.0", port=port)