#!/bin/bash

docker build -f ./Dockerfile.txt -t mytf .

docker run -t -d -p 8888:8888 --name mytf mytf

# docker exec -it mytf jupyter notebook --no-browser --ip=0.0.0.0 --allow-root --port=8888

# docker system prune -a -f