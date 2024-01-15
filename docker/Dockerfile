FROM python:3.10

# Set the working directory within the container
WORKDIR /app

RUN ls
# Copy necessary files to the container
COPY requirements.txt ./
COPY src .

# Create a virtual environment in the container
RUN python3 -m venv .venv

# Activate the virtual environment
ENV PATH="/app/.venv/bin:$PATH"

# Install Python dependencies from the requirements file
RUN pip install --no-cache-dir -r requirements.txt

# Get the models from Hugging Face to bake into the container
RUN python3 app/download_models.py

# Make port 8000 available to the world outside this container
EXPOSE 8000

# Run main.py when the container launches
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
