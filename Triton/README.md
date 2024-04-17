## Triton

### ONNX Backend
```
python3 export_pix2struct_to_single_onnx.py
docker run --rm -p8000:8000 -p8001:8001 -p8002:8002 -v/Users/naormatania/code/cyber-ai/Triton/model_repository:/models nvcr.io/nvidia/tritonserver:24.03-py3 tritonserver --model-repository=/models
```

## Python Backend
```
docker run --rm -p8000:8000 -p8001:8001 -p8002:8002 -v/Users/naormatania/code/cyber-ai/Triton/model_repository:/models -it --entrypoint bash nvcr.io/nvidia/tritonserver:24.03-py3
```
In docker container:
```
pip install numpy torch transformers pillow
tritonserver --model-repository=/models
```