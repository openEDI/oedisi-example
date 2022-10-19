FROM python:3.10.6-slim-bullseye
#USER root
RUN apt-get update && apt-get install -y git ssh

RUN mkdir -p /root/.ssh
ENV GIT_SSH_COMMAND="ssh -i /run/secrets/gadal_github_key"
RUN ssh-keyscan -t rsa github.com >> ~/.ssh/known_hosts
RUN --mount=type=secret,id=gadal_github_key pip install git+ssh://git@github.com/openEDI/GADAL@v0.2.2

WORKDIR /simulation

COPY test_full_systems.py .
COPY scenarios/docker_system.json docker_system.json
COPY LocalFeeder LocalFeeder
COPY README.md .
COPY measuring_federate measuring_federate
COPY wls_federate wls_federate
COPY recorder recorder

RUN mkdir -p outputs build
RUN python test_full_systems.py --system scenarios/docker_system.json

COPY requirements.txt .
RUN pip install -r requirements.txt
ENTRYPOINT ["helics", "run", "--path=build/test_system_runner.json"]
