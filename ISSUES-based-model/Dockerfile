# This is a sample Dockerfile
# You may adapt this to meet your environment needs

FROM python:3.10

SHELL ["/bin/bash", "-c"]
# Install necessary libraries; tesseract & dependencies for opencv
RUN apt-get update && \
    apt-get install -y tesseract-ocr
    
# Set the working directory in the container
WORKDIR /app

# Copy your Python script into the container at /app
COPY . /app/

# Install python requirements
RUN pip install -r requirements.txt

ENTRYPOINT ["python3", "process.py"]