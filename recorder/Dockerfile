FROM python:3.10.6-slim-bullseye

RUN apt-get update 
RUN apt-get install -y git ssh


RUN mkdir recorder
COPY  recorder/* ./recorder
COPY  requirements.txt ./measuring_federate/requirements.txt
WORKDIR ./recorder

RUN pip install -r requirements.txt
RUN mkdir -p outputs build

EXPOSE 8768/tcp

CMD ["python", "record_subscription.py"]

