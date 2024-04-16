FROM python:3.10.6-slim-bullseye
RUN apt-get update
RUN apt-get install -y git ssh
RUN mkdir MeasurementComponent
COPY  . ./MeasurementComponent
WORKDIR ./MeasurementComponent
RUN pip install -r requirements.txt
EXPOSE 5684/tcp
CMD ["python", "server.py"]
