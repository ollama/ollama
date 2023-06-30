FROM ubuntu:22.04
RUN apt-get update && apt-get install -y build-essential python3.11 python3-pip
RUN python3.11 -m pip install ollama
ENTRYPOINT ["ollama"]

