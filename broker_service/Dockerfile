FROM python:3.10.6-slim-bullseye

RUN apt-get update 
RUN apt-get install -y git ssh


RUN mkdir broker_service
COPY  * ./broker_service
# COPY  requirements.txt ./broker_service/requirements.txt
WORKDIR ./broker_service

RUN pip install -r requirements.txt
RUN mkdir -p outputs build

EXPOSE 8769/tcp

CMD ["python", "main.py"]

