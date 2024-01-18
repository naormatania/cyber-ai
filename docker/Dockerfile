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
RUN pip install --no-cache-dir -r requirements.txt

# Get the models from Hugging Face to bake into the container
RUN python3 download_models.py

FROM base as build

COPY src .

# Make port 8000 available to the world outside this container
EXPOSE 8000

# Run main.py when the container launches
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
