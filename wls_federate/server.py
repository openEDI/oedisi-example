from state_estimator_federate import AlgorithmParameters, StateEstimatorFederate
from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel
from typing import List, Dict
import uvicorn
import socket
import json
import sys

class StateEstimationConfig(BaseModel):
    name : str
    algorithm_parameters: AlgorithmParameters
    
class StateEstimationMapping(BaseModel):
    state_estimation_config: StateEstimationConfig
    input_mapping: Dict

app = FastAPI()

def run(state_estimation_mapping : StateEstimationMapping):
    
    config = state_estimation_mapping.state_estimation_config
    federate_name = config.name
    parameters = config.algorithm_parameters
    input_mapping = state_estimation_mapping.input_mapping
    sfed = StateEstimatorFederate(federate_name, parameters, input_mapping)
    sfed.run()

@app.get("/")
def read_root():
    hostname = socket.gethostname()
    host_ip = socket.gethostbyname(hostname)
    return {"hostname": hostname, "host ip": host_ip}


@app.post("/run/")
async def run_model(state_estimation_mapping:StateEstimationMapping, background_tasks: BackgroundTasks):
    try:
        background_tasks.add_task(run, state_estimation_mapping)
        return {"reply": "success", "error": False}
    except Exception as e:
        return {"reply": str(e), "error": True}

if __name__ == "__main__":
    port = int(sys.argv[2])
    uvicorn.run(app, host="0.0.0.0", port=port)