@echo off
chcp 65001 > nul
cd /d "%~dp0"
if not exist venv (
  echo [1/3] انشاء بيئة Python افتراضية لاول مرة...
  python -m venv venv
  echo [2/3] تثبيت المتطلبات...
  venv\Scripts\pip install -r requirements.txt
)
echo [3/3] تشغيل المنصة... افتح المتصفح على http://localhost:8000
venv\Scripts\python app.py
pause
