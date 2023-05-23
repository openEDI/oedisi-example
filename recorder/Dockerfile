FROM python:3.10.6-slim-bullseye

RUN apt-get update 
RUN apt-get install -y git ssh


RUN mkdir recorder
COPY  * ./recorder
WORKDIR ./recorder

RUN pip install -r requirements.txt

EXPOSE 8766/tcp

CMD ["python", "server.py"]

