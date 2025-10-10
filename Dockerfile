FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt /app/
RUN pip install -r requirements.txt

COPY . /app/

RUN python -c "import cv2; import numpy; print(cv2.__version__)"

EXPOSE 8080

CMD ["sh", "-c", "uvicorn pages.api:app --host 0.0.0.0 --port $PORT"]