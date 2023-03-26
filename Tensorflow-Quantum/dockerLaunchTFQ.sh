#!/bin/bash

docker build -f ./Dockerfile.txt -t mytfq .

docker run -t -d -p 8888:8888 --name mytfq mytfq

# docker exec -it mytfq jupyter notebook --no-browser --ip=0.0.0.0 --allow-root --port=8888

# docker system prune -a -f