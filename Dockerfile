FROM nvidia/cuda:12.3.2-cudnn9-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-venv python3-pip \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN python3.11 -m pip install --upgrade pip && python3.11 -m pip install -r /app/requirements.txt

COPY . /app

EXPOSE 8501
CMD ["python3.11", "-m", "streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]

