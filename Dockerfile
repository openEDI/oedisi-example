FROM python:3.10.6-slim-bullseye
#USER root
RUN apt-get update && apt-get install -y git ssh

RUN mkdir -p /root/.ssh

WORKDIR /simulation

COPY test_full_systems.py .
COPY scenarios/docker_system.json docker_system.json
COPY components.json .
COPY LocalFeeder LocalFeeder
COPY README.md .
COPY measuring_federate measuring_federate
COPY wls_federate wls_federate
COPY recorder recorder

RUN mkdir -p outputs build

COPY requirements.txt .
RUN pip install -r requirements.txt

RUN gadal build --system docker_system.json
ENTRYPOINT ["gadal", "run"]
