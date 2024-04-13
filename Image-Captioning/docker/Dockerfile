FROM python:3.10 as base

# Set the working directory within the container
WORKDIR /app

COPY requirements.txt ./
COPY build .

# Create a virtual environment in the container
RUN python3 -m venv .venv

# Activate the virtual environment
ENV PATH="/app/.venv/bin:$PATH"

# Install Python dependencies from the requirements file
RUN --mount=type=cache,target=/root/.cache pip install -r requirements.txt

# Get the models from Hugging Face to bake into the container
RUN mkdir -p models
COPY models models

# Get the exported ONNX models into the container
RUN mkdir -p onnx
COPY build/onnx onnx

# Get the exported ONNX models into the container
# RUN mkdir -p onnx
# COPY build/onnx onnx

FROM base as build

COPY src .

# Make port 8000 available to the world outside this container
EXPOSE 8000

ENV TQDM_DISABLE 1

ARG INTRAOP_THREADS
ARG INTEROP_THREADS
ARG OPTIMIZATION
ARG BATCH_SIZE

ENV INTRAOP_THREADS ${INTRAOP_THREADS}
ENV INTEROP_THREADS ${INTEROP_THREADS}
ENV OPTIMIZATION ${OPTIMIZATION}
ENV BATCH_SIZE ${BATCH_SIZE}

# ENV TORCH_LOGS "+dynamo"
# ENV TORCHDYNAMO_VERBOSE 1

# Run main.py when the container launches
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
