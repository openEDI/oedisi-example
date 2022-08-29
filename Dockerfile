FROM python:3.10.6-slim-bullseye
#USER root
RUN apt-get update && apt-get install -y git ssh

RUN mkdir -p /root/.ssh
ENV GIT_SSH_COMMAND="ssh -i /run/secrets/gadal_github_key"
RUN ssh-keyscan -t rsa github.com >> ~/.ssh/known_hosts
RUN --mount=type=secret,id=gadal_github_key sha256sum /run/secrets/gadal_github_key && wc -l /run/secrets/gadal_github_key
RUN --mount=type=secret,id=gadal_github_key pip install git+ssh://git@github.com/openEDI/GADAL@v0.2.1

WORKDIR /simulation

COPY test_full_systems.py .
COPY test_system.json .
COPY AWSFeeder AWSFeeder
COPY LocalFeeder LocalFeeder
COPY README.md .
COPY measuring_federate measuring_federate
COPY wls_federate wls_federate
COPY recorder recorder
RUN python test_full_systems.py

COPY requirements.txt .
RUN pip install -r requirements.txt
ENTRYPOINT ["helics", "run", "--path=test_system_runner.json"]
