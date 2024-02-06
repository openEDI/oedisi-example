FROM python:3.10.6-slim-bullseye
#USER root
RUN apt-get update && apt-get install -y git ssh

RUN mkdir -p /root/.ssh

WORKDIR /simulation

COPY scenarios/docker_system.json docker_system.json
COPY components.json .
COPY LocalFeeder LocalFeeder
COPY lindistflow_federate lindistflow_federate
COPY README.md .
COPY measuring_federate measuring_federate
COPY wls_federate wls_federate
COPY recorder recorder
COPY omoo_federate omoo_federate

RUN mkdir -p outputs build

COPY requirements.txt .
RUN pip install -r requirements.txt

RUN oedisi build --system docker_system.json
ENTRYPOINT ["oedisi", "run"]
