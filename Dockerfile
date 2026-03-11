FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .
COPY generate_weekly_history.py .
COPY manage_inventory.py .
COPY shortage_forecast.py .
COPY train_shortage_model.py .
COPY update_inventory.py .
COPY templates/ templates/
COPY models/ models/
COPY medicine_inventory_dummy_data_v2/ medicine_inventory_dummy_data_v2/

EXPOSE 8080

CMD ["gunicorn", "--bind", "0.0.0.0:8080", "--workers", "1", "--threads", "8", "--timeout", "120", "app:app"]
