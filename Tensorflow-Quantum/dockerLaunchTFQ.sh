#!/bin/bash

# Docker build and run

docker build -f ./Dockerfile.txt -t mytfq .

docker run -it -p 8888:8888 mytfq

# Docker clean up

# docker system prune -a -f