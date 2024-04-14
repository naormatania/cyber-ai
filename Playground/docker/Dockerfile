FROM python:3.10 as base

WORKDIR /app

COPY Playground/requirements.txt ./

RUN mkdir -p external_deps
COPY NER/external_deps external_deps

RUN --mount=type=cache,target=/root/.cache pip install -r requirements.txt

RUN mkdir -p onnx
COPY NER/build/onnx/SecureBERT_NER onnx/SecureBERT_NER
COPY Image-Captioning/build/onnx/pix2struct_base onnx/pix2struct_base

COPY Image-Captioning/src/app/cache_model.py .
RUN python cache_model.py

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
