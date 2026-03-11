# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY Value/requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application files into the container
COPY Value/app.py .
COPY Value/generate_weekly_history.py .
COPY Value/manage_inventory.py .
COPY Value/shortage_forecast.py .
COPY Value/train_shortage_model.py .
COPY Value/update_inventory.py .
COPY .gitignore .

# Expose the port that Cloud Run will set via environment variable
EXPOSE 8080

# Run app.py when the container launches, using the PORT env var
CMD ["sh", "-c", "python app.py --port=$PORT"]