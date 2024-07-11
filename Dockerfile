# Dockerfile

# Use an official Python runtime as a parent image
FROM python:3.12-slim

# Set the working directory in the container
WORKDIR /app

# Install any needed packages specified in requirements.txt
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Install Tesseract and additional language data
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    tesseract-ocr-ces

# Copy the local script.py to the container
COPY foto_script.py /app/

# Define the command to run your script
CMD ["python3", "foto_script.py"]
