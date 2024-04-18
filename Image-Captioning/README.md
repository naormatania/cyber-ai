# image-captioning

## Run the container
```
DOCKER_BUILDKIT=1 docker build --build-arg INTRAOP_THREADS=4 --build-arg INTEROP_THREADS=4 --build-arg BATCH_SIZE=4 -t localbuild:caption_service --target build -f docker/Dockerfile . && docker run -p 8000:8000 -it localbuild:caption_service
```

## Test
With docker container up:
```
python3 test/main.py blip --batch_size=12
```

## Run Streamlit
```
DOCKER_BUILDKIT=1 docker build --target base -f docker/Dockerfile.streamlit .
DOCKER_BUILDKIT=1 docker build -t localbuild:streamlit --target build -f docker/Dockerfile.streamlit . && docker run -p 8501:8501 -it localbuild:streamlit
```