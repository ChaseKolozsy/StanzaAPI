# Stage 1: Build dependencies
FROM python:3.10-alpine as builder

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && \
    rm requirements.txt

# Stage 2: Final lightweight image
FROM python:3.10-slim

WORKDIR /app

COPY --from=builder /app /app

COPY app.py .

CMD ["gunicorn", "--bind", "0.0.0.0:5004", "app:app"]