FROM pytorch/pytorch:1.9.0-cuda11.1-cudnn8-runtime

RUN apt update && apt install -y python3-dev python3-pip

RUN mkdir /app
COPY ./requirements.txt /app/requirements.txt
COPY ./app /app
COPY ./config /config

WORKDIR /app
RUN python3 -m pip install -r requirements.txt

EXPOSE 8000

CMD ["gunicorn", "-c", "/config/gunicorn.py", "-k", "uvicorn.workers.UvicornWorker", "main:app"]
