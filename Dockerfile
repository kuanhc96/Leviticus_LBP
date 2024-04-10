FROM python:3.9-slim-bullseye

WORKDIR /app

COPY requirements.txt /app/

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6 -y

RUN pip install --progress-bar off --no-cache-dir -r requirements.txt

COPY . /app/

VOLUME ./texture_dataset /app/texture_dataset

EXPOSE 8000

CMD ["uvicorn", "test_fastapi:app", "--host", "0.0.0.0", "--port", "8000"]