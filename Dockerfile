# syntax=docker/dockerfile:1

FROM node:latest
WORKDIR /app

ARG OLLAMA_API_ENDPOINT=''
RUN echo $OLLAMA_API_ENDPOINT

ENV ENV prod

ENV PUBLIC_API_ENDPOINT $OLLAMA_API_ENDPOINT
RUN echo $PUBLIC_API_ENDPOINT

COPY package.json package-lock.json ./ 
RUN npm ci

COPY . .
RUN npm run build

CMD [ "npm", "run", "start"]
