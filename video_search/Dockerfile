# Use the official Python runtime as a parent image
FROM python:3.10

# Set the working directory in the container
WORKDIR /app

# Copy only the necessary files
COPY ./src /app/src
COPY ./pyproject.toml /app

# Install poetry to manage dependencies
RUN pip install --no-cache-dir poetry

# Install dependencies
RUN poetry config virtualenvs.create false && \
    poetry install --no-dev

# Expose the FastAPI port
EXPOSE 8000

# CMD to run the FastAPI application with Uvicorn
CMD ["uvicorn", "src.video_detector:app", "--host", "0.0.0.0", "--port", "8000"]
