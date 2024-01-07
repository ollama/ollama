# syntax=docker/dockerfile:1

FROM node:alpine as build

WORKDIR /app

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
COPY --from=build /app/build /app/build

WORKDIR /app/backend

COPY ./backend/requirements.txt ./requirements.txt
RUN pip3 install -r requirements.txt
# RUN python -c "from sentence_transformers import SentenceTransformer; model = SentenceTransformer('all-MiniLM-L6-v2')"

COPY ./backend .

CMD [ "sh", "start.sh"]