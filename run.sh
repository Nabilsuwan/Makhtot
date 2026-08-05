#!/usr/bin/env bash
cd "$(dirname "$0")"
if [ ! -d venv ]; then
  echo "[1/3] إنشاء بيئة Python افتراضية لأول مرة..."
  python3 -m venv venv
  echo "[2/3] تثبيت المتطلبات..."
  venv/bin/pip install -r requirements.txt
fi
echo "[3/3] تشغيل المنصة... افتح المتصفح على http://localhost:8000"
venv/bin/python app.py
