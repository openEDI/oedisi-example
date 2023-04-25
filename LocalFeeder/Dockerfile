FROM python:3.10.6-slim-bullseye

RUN apt-get update 
RUN apt-get install -y git ssh


RUN mkdir LocalFeeder
COPY  LocalFeeder/* ./LocalFeeder
COPY  requirements.txt ./LocalFeeder/requirements.txt
WORKDIR ./LocalFeeder

RUN pip install -r requirements.txt
RUN mkdir -p outputs build

EXPOSE 8765/tcp

CMD ["python", "sender_cosim.py"]

