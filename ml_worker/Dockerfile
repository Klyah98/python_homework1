FROM python:3.8.12

RUN mkdir -p /app

ENV MONGO_DB_PORT 27017
ENV MONGO_DB_HOST mongo_db
ENV CELERY_BROKER redis://redis:6379/0
ENV CELERY_BACKEND redis://redis:6379/0

COPY . /app

WORKDIR /app

RUN pip3 install -r requirements.txt

ENTRYPOINT celery -A tasks worker --loglevel=info
