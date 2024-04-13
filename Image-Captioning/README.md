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
git clone git@github.com:naormatania/optimum.git
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

## Triton
```
cd docs/examples
./fetch_models.sh
```
and save in `/full/path/to/docs/examples/model_repository`
and then run:
```
docker run --rm -p8000:8000 -p8001:8001 -p8002:8002 -v/full/path/to/docs/examples/model_repository:/models nvcr.io/nvidia/tritonserver:24.03-py3 tritonserver --model-repository=/models
pip install tritonclient[all]
```
Relevant example for usage: https://github.com/triton-inference-server/client/blob/main/src/python/examples/image_client.py