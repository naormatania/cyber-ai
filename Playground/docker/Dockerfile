FROM python:3.10 as base

WORKDIR /app

COPY Playground/requirements.txt ./

RUN mkdir -p external_deps
COPY NER/external_deps external_deps

RUN --mount=type=cache,target=/root/.cache pip install -r requirements.txt

RUN mkdir -p onnx
COPY NER/build/onnx/SecureBERT_NER onnx/SecureBERT_NER
COPY Image-Captioning/build/onnx/pix2struct_base onnx/pix2struct_base

RUN mkdir -p models
COPY Image-Captioning/models/lavis-blip-base models/lavis-blip-base
COPY Image-Captioning/build/blip_caption_base_coco.yaml models/lavis-blip-base
COPY Image-Captioning/build/med_config.json models/lavis-blip-base

# TODO: to prevent network access during build - run script before docker build and copy ~/.cache/huggingface/hub/models--bert-base-uncased inside
COPY Image-Captioning/build/cache_tokenizer.py .
RUN python cache_tokenizer.py

FROM base as build

COPY NER/src .
COPY Playground/src/main.py .

EXPOSE 8501

ENV TQDM_DISABLE 1
ENV STREAMLIT_SERVER_MAX_UPLOAD_SIZE 100

ENV INTRAOP_THREADS 4
ENV INTEROP_THREADS 4

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["streamlit", "run", "main.py", "--server.port=8501", "--server.address=0.0.0.0"]
