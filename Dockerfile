# syntax=docker/dockerfile:1

FROM node:alpine as build

WORKDIR /app

# wget embedding model weight from alpine (does not exist from slim-buster)
RUN wget "https://chroma-onnx-models.s3.amazonaws.com/all-MiniLM-L6-v2/onnx.tar.gz"

COPY package.json package-lock.json ./ 
RUN npm ci

COPY . .
RUN npm run build


FROM python:3.11-slim-bookworm as base

ENV ENV=prod

ENV OLLAMA_API_BASE_URL "/ollama/api"

ENV OPENAI_API_BASE_URL ""
ENV OPENAI_API_KEY ""

ENV WEBUI_JWT_SECRET_KEY "SECRET_KEY"

WORKDIR /app

# copy embedding weight from build
RUN mkdir -p /root/.cache/chroma/onnx_models/all-MiniLM-L6-v2
COPY --from=build /app/onnx.tar.gz /root/.cache/chroma/onnx_models/all-MiniLM-L6-v2

RUN cd /root/.cache/chroma/onnx_models/all-MiniLM-L6-v2 &&\
    tar -xzf onnx.tar.gz

# copy built frontend files
COPY --from=build /app/build /app/build

WORKDIR /app/backend

COPY ./backend/requirements.txt ./requirements.txt
RUN pip3 install -r requirements.txt

# RUN python -c "from sentence_transformers import SentenceTransformer; model = SentenceTransformer('all-MiniLM-L6-v2')"

COPY ./backend .

CMD [ "sh", "start.sh"]