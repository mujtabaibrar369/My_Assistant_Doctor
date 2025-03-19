# Use an official Python runtime as the base image
FROM python:3.9

# Set the working directory inside the container
WORKDIR /app

# Copy all files from your project directory to the container
COPY . /app

# Install dependencies
RUN pip install --no-cache-dir --default-timeout=100 -r requirements.txt

# Expose the Flask port (default is 5000)
EXPOSE 5000

# Define environment variable for Flask
ENV FLASK_APP=app.py

# Run the Flask application
CMD ["python", "app.py"]
