# cyber-ner

## Run the container
```
docker build -t localbuild:ner_service -f docker/Dockerfile .
docker run -it localbuild:ner_service
```

## Query the container
```
curl http://localhost:6000/ner/cyner/ POST -H "Content-Type: application/json" -v -d '{"text": "Proofpoint report mentions that the German-language messages were turned off once the UK messages were established, indicating a conscious effort to spread FluBot 446833e3f8b04d4c3c2d2288e456328266524e396adbfeba3769d00727481e80 in Android phones."}'
```