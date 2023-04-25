FROM python:3.10.6-slim-bullseye

RUN apt-get update 
RUN apt-get install -y git ssh


RUN mkdir wls_federate
COPY  wls_federate/* ./wls_federate
COPY  requirements.txt ./measuring_federate/requirements.txt
WORKDIR ./wls_federate

RUN pip install -r requirements.txt
RUN mkdir -p outputs build

EXPOSE 8767/tcp

CMD ["python", "state_estimator_federate.py"]

