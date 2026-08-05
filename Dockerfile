FROM python:3.12-slim

# Tesseract العربي مضمَّن — لا حاجة لأي تثبيت يدوي على السحابة
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr tesseract-ocr-ara libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .

ENV HOST=0.0.0.0
ENV PORT=8000
ENV DATA_DIR=/app/data
EXPOSE 8000
CMD ["python", "app.py"]
