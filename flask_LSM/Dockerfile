# Use the official lightweight Python image.
# https://hub.docker.com/_/python
# Base image
FROM python:3.9-slim-buster

# Set working directory
WORKDIR /app

# Copy the requirements.txt file to the working directory
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire current directory to the working directory
COPY . .

# Expose port 8080
EXPOSE 8080

# Set the environment variables
ENV CAMERA_DEVICE_INDEX=0

# Start the application
CMD ["python", "flask_test.py"]
