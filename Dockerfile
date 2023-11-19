# syntax=docker/dockerfile:1

FROM node:alpine as build

ARG OLLAMA_API_BASE_URL='/ollama/api'
RUN echo $OLLAMA_API_BASE_URL

ENV PUBLIC_API_BASE_URL $OLLAMA_API_BASE_URL
RUN echo $PUBLIC_API_BASE_URL

WORKDIR /app

COPY package.json package-lock.json ./ 
RUN npm ci

COPY . .
RUN npm run build

FROM python:3.11-slim-buster as base

ARG OLLAMA_API_BASE_URL='/ollama/api'

ENV ENV=prod
ENV OLLAMA_API_BASE_URL $OLLAMA_API_BASE_URL
ENV WEBUI_AUTH ""
ENV WEBUI_DB_URL ""
ENV WEBUI_JWT_SECRET_KEY "SECRET_KEY"

WORKDIR /app
COPY --from=build /app/build /app/build

WORKDIR /app/backend

COPY ./backend/requirements.txt ./requirements.txt
RUN pip3 install -r requirements.txt

COPY ./backend .

CMD [ "sh", "start.sh"]