from fastapi import FastAPI, BackgroundTasks
from gadal.gadal_types.data_types import MeasurementArray
from measuring_federate import MeasurementMapping
import uvicorn
import socket
import sys

app = FastAPI()

def run(model_configs : MeasurementMapping):
    """Load static_inputs and input_mapping and run JSON."""
    config = model_configs.static_inputs
    input_mapping = model_configs.input_mapping
    sfed = MeasurementRelay(config, input_mapping)
    sfed.run()

@app.get("/")
def read_root():
    hostname = socket.gethostname()
    host_ip = socket.gethostbyname(hostname)
    return {"hostname": hostname, "host ip": host_ip}


@app.post("/run/")
async def run_model(measurement_mapping:MeasurementMapping, background_tasks: BackgroundTasks):
    try:
        background_tasks.add_task(run, measurement_mapping)
        return {"reply": "success", "error": False}
    except Exception as e:
        return {"reply": str(e), "error": True}

if __name__ == "__main__":
    port = int(sys.argv[2])
    uvicorn.run(app, host="0.0.0.0", port=port)