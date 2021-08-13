FROM nvidia/cuda:11.0.3-cudnn8-runtime-ubuntu20.04

RUN apt update && apt install -y python3-dev python3-pip

COPY ./app /app
COPY ./requirements.txt /app/requirements.txt

WORKDIR /app
RUN python3 -m pip install -r requirements.txt

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
