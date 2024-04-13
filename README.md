# AI for Cyber purposes

```
DOCKER_BUILDKIT=1 docker build -t localbuild:streamlit --target build -f Playground/docker/Dockerfile . && docker run -p 8501:8501 -it localbuild:streamlit
```

1. NER - extract cyber entities
2. Image-Captioning - caption desktop, screenshots and regular images
3. Captioning-Search - Search over image captions