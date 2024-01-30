# syntax=docker/dockerfile:1

FROM node:alpine as build

WORKDIR /app

# wget embedding model weight from alpine (does not exist from slim-buster)
RUN wget "https://chroma-onnx-models.s3.amazonaws.com/all-MiniLM-L6-v2/onnx.tar.gz" -O - | \
    tar -xzf - -C /app

COPY package.json package-lock.json ./
RUN npm ci

COPY . .
RUN npm run build


FROM python:3.11-slim-bookworm as base

ENV ENV=prod
ENV PORT ""

ENV OLLAMA_API_BASE_URL "/ollama/api"

ENV OPENAI_API_BASE_URL ""
ENV OPENAI_API_KEY ""

ENV WEBUI_JWT_SECRET_KEY "SECRET_KEY"

WORKDIR /app/backend

# install python dependencies
COPY ./backend/requirements.txt ./requirements.txt

RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu --no-cache-dir
RUN pip3 install -r requirements.txt --no-cache-dir

# Install pandoc and netcat
# RUN python -c "import pypandoc; pypandoc.download_pandoc()"
RUN apt-get update \
    && apt-get install -y pandoc netcat-openbsd \
    && rm -rf /var/lib/apt/lists/*

# RUN python -c "from sentence_transformers import SentenceTransformer; model = SentenceTransformer('all-MiniLM-L6-v2')"

# copy embedding weight from build
RUN mkdir -p /root/.cache/chroma/onnx_models/all-MiniLM-L6-v2
COPY --from=build /app/onnx /root/.cache/chroma/onnx_models/all-MiniLM-L6-v2/onnx

# copy built frontend files
COPY --from=build /app/build /app/build

# copy backend files
COPY ./backend .

CMD [ "bash", "start.sh"]