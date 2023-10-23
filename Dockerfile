# syntax=docker/dockerfile:1

FROM node:alpine
WORKDIR /app

ARG OLLAMA_API_BASE_URL=''
RUN echo $OLLAMA_API_BASE_URL

ENV ENV prod

ENV PUBLIC_API_BASE_URL $OLLAMA_API_BASE_URL
RUN echo $PUBLIC_API_BASE_URL

COPY package.json package-lock.json ./ 
RUN npm ci

COPY . .
RUN npm run build

CMD [ "npm", "run", "start"]
