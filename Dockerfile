# syntax=docker/dockerfile:1

FROM oven/bun:alpine as build

WORKDIR /app

COPY package.json package-lock.json ./ 

COPY . .
RUN bun install
RUN bun run build

FROM python:3.11-slim-buster as base

ARG OLLAMA_API_BASE_URL='/ollama/api'

ENV ENV=prod
ENV OLLAMA_API_BASE_URL $OLLAMA_API_BASE_URL

ENV OPENAI_API_BASE_URL ""
ENV OPENAI_API_KEY ""

ENV WEBUI_JWT_SECRET_KEY "SECRET_KEY"

WORKDIR /app
COPY --from=build /app/build /app/build

WORKDIR /app/backend

COPY ./backend/requirements.txt ./requirements.txt
RUN pip3 install -r requirements.txt

COPY ./backend .

CMD [ "sh", "start.sh"]