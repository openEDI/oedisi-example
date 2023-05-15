FROM python:3.10.6-slim-bullseye

RUN apt-get update 
RUN apt-get install -y git ssh


RUN mkdir measuring_federate
COPY  * ./measuring_federate
#COPY  requirements.txt ./measuring_federate/requirements.txt
WORKDIR ./measuring_federate

RUN pip install -r requirements.txt
RUN mkdir -p outputs build

EXPOSE 8766/tcp

CMD ["python", "measuring_federate.py"]

