# syntax=docker/dockerfile:1

FROM node:latest

WORKDIR /app
ENV ENV prod

COPY package.json package-lock.json ./ 
RUN npm ci


COPY . .
RUN npm run build

CMD [ "node", "./build/index.js"]
