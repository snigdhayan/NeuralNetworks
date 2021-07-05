#!/bin/bash

docker build -f ./Dockerfile.txt -t myfastai .

docker run -t -d -p 8888:8888 --name myfastai myfastai

docker exec -it myfastai jupyter notebook --no-browser --ip=0.0.0.0 --allow-root --port=8888

# docker system prune -a -f