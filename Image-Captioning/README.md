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

## Export pix2struct as onnx
In `build/`:
```
git clone git@github.com:huggingface/optimum.git
```
In `build/optimum`:
```
python3 setup.py develop
cp /usr/local/bin/optimum-cli .
```
In `build`:
```
./optimum/optimum-cli export onnx -m google/pix2struct-screen2words-base --optimize O3 onnx/pix2struct_base
```