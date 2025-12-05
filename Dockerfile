# 1. Use the specific Python version you requested
FROM python:3.10.19

# 2. Set the working directory inside the container
WORKDIR /code

# 3. Copy requirements first (for caching)
COPY ./requirements.txt /code/requirements.txt

# 4. Install libraries normally (Standard PyTorch will be installed)
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# 5. Copy your source code and data
COPY ./app /code/app
COPY ./data /code/data

# 6. Create a cache directory for the AI model
# Hugging Face runs as a non-root user, so we set permissions to 777
RUN mkdir -p /code/model_cache && chmod 777 /code/model_cache

# 7. Start the server on port 7860 (Required for Hugging Face)
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]