from pydantic import BaseModel

class Broker(BaseModel):
    num_federates : int = 1
    debug: bool = False
    broker_name : str = "mainbroker"
    broker_port : int  = 23404
    broker_ip: str = "0.0.0.0"
    core_type: str = "zmq"
        