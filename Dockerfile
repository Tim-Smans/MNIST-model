# Use the python image
FROM python:3.9

# Set working directory
WORKDIR /app

# Copy the files to the container
COPY models/mnist_model.pth mnist_model.pth
COPY app/app.py app.py
COPY requirements.txt requirements.txt

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose port 5000 for API
EXPOSE 5000

# Run the app
CMD ["python", "app.py"]
