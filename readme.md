# Background
This repository contains infrastructure (ELK stack) and HTTP server to support machine learning inference.

This repository is built on top of amazing tools, in no particular order:
1. [FastAPI](https://github.com/tiangolo/fastapi)
2. [Elastic stack (ELK) on Docker](https://github.com/deviantony/docker-elk)
3. [Transformers by Hugging Face](https://github.com/huggingface/transformers)

# Modifications
You must rebuild the stack images with `docker-compose build` (or `docker-compose up -d --build`) whenever you switch branch, updating the version, or editing the code on an already existing stack.

# Installations

## Filebeat
Installing Filebeat is performed by following the [official quick start guideline](https://www.elastic.co/guide/en/beats/filebeat/7.10/filebeat-installation-configuration.html). The steps are:
1. SSH to the honeypot node
2. Install the .deb file
3. Use the provided [config file](filebeat/config/filebeat.yml) as the Filebeat's configuration
4. Adjust log path and Logstash address accordingly

## ELK
Elasticsearch, Logstash and Kibana is installed with ELK on Docker. The configuration for the services are available in respective folders.
