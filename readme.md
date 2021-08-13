# Background
This repository contains infrastructure (ELK stack) and HTTP server to support machine learning inference.

This repository is built on top of amazing tools, in no particular order:
1. [FastAPI](https://github.com/tiangolo/fastapi)
2. [Elastic stack (ELK) on Docker](https://github.com/deviantony/docker-elk)

# Modifications
You must rebuild the stack images with `docker-compose build` whenever you switch branch, updating the version, or editing the code on an already existing stack.