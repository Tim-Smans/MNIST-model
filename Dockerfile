# Use an official Python image
FROM python:3.9

# Set the working directory
WORKDIR /app

# Copy necessary files to the container
COPY models/mnist_model.pth mnist_model.pth
COPY app/app.py app.py
COPY requirements.txt requirements.txt

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose port 5000 for API
EXPOSE 5000

# Run the app
CMD ["python", "app.py"]
