# cyber-ner

## Run the container
```
DOCKER_BUILDKIT=1 docker build --target base -f docker/Dockerfile .
    DOCKER_BUILDKIT=1 docker build --build-arg INTRAOP_THREADS=1 --build-arg INTEROP_THREADS=1 --build-arg OPTIMIZATION=BETTER_TRANSFORMER -t localbuild:ner_service --target build -f docker/Dockerfile . && docker run -p 8000:8000 -it localbuild:ner_service
```

## Query the container

CyNER:
```
curl http://127.0.0.1:8000/ner/cyner/ -H "Content-Type: application/json" -v -d '{"text": "Proofpoint report mentions that the German-language messages were turned off once the UK messages were established, indicating a conscious effort to spread FluBot 446833e3f8b04d4c3c2d2288e456328266524e396adbfeba3769d00727481e80 in Android phones."}'
```

SecureBERT-NER:
```
curl http://127.0.0.1:8000/ner/secner/ -H "Content-Type: application/json" -v -d '{"text": "Proofpoint report mentions that the German-language messages were turned off once the UK messages were established, indicating a conscious effort to spread FluBot 446833e3f8b04d4c3c2d2288e456328266524e396adbfeba3769d00727481e80 in Android phones."}'
```

## Performance test
With docker container up:
```
python3 test/main.py secner --num_sentences=10
```

## Run Streamlit
```
DOCKER_BUILDKIT=1 docker build --target base -f docker/Dockerfile.streamlit .
DOCKER_BUILDKIT=1 docker build -t localbuild:streamlit --target build -f docker/Dockerfile.streamlit . && docker run -p 8501:8501 -it localbuild:streamlit
```