"""
منصة رقمنة المخطوطات — النسخة المستقلة (بدون Docker)
=====================================================
كل شيء في عملية Python واحدة:
  - SQLite بدل PostgreSQL (لا تثبيت قاعدة بيانات)
  - معالجة خلفية بخيط مدمج بدل Celery/Redis
  - واجهة ويب HTML مدمجة يقدّمها الخادم نفسه بدل React/Node
  - نفس خط أنابيب المعالجة المُختبَر على مخطوطة حقيقية

التشغيل:  python app.py   ثم افتح  http://localhost:8000
المتطلب النظامي الوحيد الاختياري: tesseract-ocr + tesseract-ocr-ara
(إن لم يكن مثبتًا، تعمل المنصة ويُعطَّل OCR مع رسالة توضح كيفية تثبيته)
"""
from __future__ import annotations

import hashlib
import re
import shutil
import threading
import uuid
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from fastapi import BackgroundTasks, FastAPI, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from sqlalchemy import (
    Boolean, DateTime, Float, ForeignKey, Integer, String, Text, create_engine, select,
)
from sqlalchemy.orm import (
    DeclarativeBase, Mapped, Session, mapped_column, relationship, sessionmaker,
)

# ---------------------------------------------------------------- الإعدادات
import os

BASE_DIR = Path(__file__).resolve().parent
# DATA_DIR قابل للضبط عبر متغير بيئة: في السحابة اربطه بقرص دائم (Volume)
# حتى لا تُمسح البيانات عند إعادة النشر. محليًا يبقى بجوار البرنامج.
DATA_DIR = Path(os.environ.get("DATA_DIR", BASE_DIR / "data"))
ARCHIVE_DIR = DATA_DIR / "archive"      # الأصول المحفوظة (لا تُعدَّل أبدًا)
DERIVED_DIR = DATA_DIR / "derived"      # الصور المعالَجة
DB_PATH = DATA_DIR / "manuscripts.db"
REVIEW_THRESHOLD = 70.0                 # أي كلمة أقل من هذه الثقة تدخل المراجعة

for d in (ARCHIVE_DIR, DERIVED_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------ قاعدة البيانات
engine = create_engine(f"sqlite:///{DB_PATH}", connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine, autoflush=False)


class Base(DeclarativeBase):
    pass


class Manuscript(Base):
    __tablename__ = "manuscripts"
    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    title: Mapped[str] = mapped_column(String(500))
    call_number: Mapped[str | None] = mapped_column(String(100), nullable=True)
    original_path: Mapped[str] = mapped_column(String(1000))
    checksum_sha256: Mapped[str] = mapped_column(String(64))
    page_count: Mapped[int] = mapped_column(Integer, default=0)
    status: Mapped[str] = mapped_column(String(50), default="uploaded")
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    uploaded_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    pages: Mapped[list["Page"]] = relationship(back_populates="manuscript")


class Page(Base):
    __tablename__ = "pages"
    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    manuscript_id: Mapped[str] = mapped_column(ForeignKey("manuscripts.id"))
    page_number: Mapped[int] = mapped_column(Integer)
    original_image_path: Mapped[str] = mapped_column(String(1000))
    processed_image_path: Mapped[str | None] = mapped_column(String(1000), nullable=True)
    full_text: Mapped[str] = mapped_column(Text, default="")
    normalized_text: Mapped[str] = mapped_column(Text, default="")  # للبحث المرن
    avg_confidence: Mapped[float] = mapped_column(Float, default=0.0)
    manuscript: Mapped[Manuscript] = relationship(back_populates="pages")
    words: Mapped[list["Word"]] = relationship(back_populates="page")


class Word(Base):
    __tablename__ = "words"
    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    page_id: Mapped[str] = mapped_column(ForeignKey("pages.id"))
    text: Mapped[str] = mapped_column(String(200))
    confidence: Mapped[float] = mapped_column(Float)
    x: Mapped[int] = mapped_column(Integer)
    y: Mapped[int] = mapped_column(Integer)
    w: Mapped[int] = mapped_column(Integer)
    h: Mapped[int] = mapped_column(Integer)
    corrected_text: Mapped[str | None] = mapped_column(String(200), nullable=True)
    is_training_sample: Mapped[bool] = mapped_column(Boolean, default=False)
    page: Mapped[Page] = relationship(back_populates="words")


Base.metadata.create_all(engine)

# ------------------------------------------------- تحسين الصورة (مُختبَر فعليًا)
def deskew(gray: np.ndarray) -> tuple[np.ndarray, float]:
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 200, minLineLength=100, maxLineGap=10)
    angles = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            a = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            if -20 < a < 20:
                angles.append(a)
    if not angles:
        return gray, 0.0
    angle = float(np.median(angles))
    h, w = gray.shape[:2]
    m = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    return cv2.warpAffine(gray, m, (w, h), flags=cv2.INTER_CUBIC,
                          borderMode=cv2.BORDER_REPLICATE), angle


def enhance(gray: np.ndarray) -> np.ndarray:
    den = cv2.fastNlMeansDenoising(gray, h=10, templateWindowSize=7, searchWindowSize=21)
    contrast = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(den)
    return cv2.adaptiveThreshold(contrast, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY, 35, 15)


# ----------------------------------------------------------- OCR (اختياري)
def tesseract_available() -> bool:
    return shutil.which("tesseract") is not None


def ocr_page(binarized: np.ndarray) -> tuple[str, float, list[dict]]:
    """يعيد (النص الكامل، متوسط الثقة، قائمة الكلمات بإحداثياتها)."""
    import pytesseract
    from pytesseract import Output

    data = pytesseract.image_to_data(binarized, lang="ara", output_type=Output.DICT,
                                     config="--psm 6")
    words, lines = [], {}
    for i in range(len(data["text"])):
        t, c = data["text"][i].strip(), float(data["conf"][i])
        if not t or c < 0:
            continue
        key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
        lines.setdefault(key, []).append(t)
        words.append({"text": t, "confidence": c, "x": data["left"][i],
                      "y": data["top"][i], "w": data["width"][i], "h": data["height"][i]})
    full_text = "\n".join(" ".join(ws) for ws in lines.values())
    avg = sum(w["confidence"] for w in words) / len(words) if words else 0.0
    return full_text, avg, words


# --------------------------------------------------- تطبيع عربي للبحث المرن
_DIACRITICS = re.compile(r"[\u064B-\u065F\u0670]")

def normalize_arabic(text: str) -> str:
    text = _DIACRITICS.sub("", text)
    for src, dst in (("أ", "ا"), ("إ", "ا"), ("آ", "ا"), ("ى", "ي"), ("ة", "ه"), ("ـ", "")):
        text = text.replace(src, dst)
    return text


# ------------------------------------------------------- خط المعالجة الخلفي
def process_manuscript_job(manuscript_id: str) -> None:
    """يعمل في خيط خلفي: استخراج صفحات → تحسين → OCR → حفظ. مع تحديث الحالة."""
    db = SessionLocal()
    try:
        m = db.get(Manuscript, manuscript_id)
        m.status = "processing"
        db.commit()

        original = Path(m.original_path)
        pages_dir = DERIVED_DIR / manuscript_id
        pages_dir.mkdir(parents=True, exist_ok=True)

        # 1) استخراج الصفحات (PDF: الصورة المضمَّنة بدقتها الأصلية الكاملة)
        page_files: list[Path] = []
        if original.suffix.lower() == ".pdf":
            import fitz
            doc = fitz.open(original)
            for i in range(doc.page_count):
                imgs = doc[i].get_images(full=True)
                if imgs:
                    base = doc.extract_image(imgs[0][0])
                    p = pages_dir / f"page_{i:04d}.{base['ext']}"
                    p.write_bytes(base["image"])
                else:  # صفحة بلا صورة مضمَّنة: تُرسم بدقة عالية كاحتياط
                    pix = doc[i].get_pixmap(dpi=300)
                    p = pages_dir / f"page_{i:04d}.png"
                    pix.save(str(p))
                page_files.append(p)
        else:
            page_files = [original]

        has_ocr = tesseract_available()

        # 2) لكل صفحة: تحسين + OCR
        for n, pf in enumerate(page_files, start=1):
            img = cv2.imread(str(pf))
            if img is None:
                continue
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            desk, _ = deskew(gray)
            binarized = enhance(desk)
            proc_path = pages_dir / f"page_{n:04d}_bin.png"
            cv2.imwrite(str(proc_path), binarized)

            page = Page(id=str(uuid.uuid4()), manuscript_id=manuscript_id,
                        page_number=n, original_image_path=str(pf),
                        processed_image_path=str(proc_path))
            db.add(page)
            db.flush()

            if has_ocr:
                full_text, avg, words = ocr_page(binarized)
                page.full_text = full_text
                page.normalized_text = normalize_arabic(full_text)
                page.avg_confidence = avg
                for w in words:
                    db.add(Word(id=str(uuid.uuid4()), page_id=page.id, text=w["text"],
                                confidence=w["confidence"], x=w["x"], y=w["y"],
                                w=w["w"], h=w["h"]))
            m.page_count = n
            db.commit()  # حفظ تدريجي: التقدم مرئي أثناء المعالجة

        m.status = "complete" if has_ocr else "complete_no_ocr"
        db.commit()
    except Exception as exc:  # noqa: BLE001 — نسجل الخطأ للمستخدم بدل الانهيار الصامت
        db.rollback()
        m = db.get(Manuscript, manuscript_id)
        if m:
            m.status = "error"
            m.error_message = str(exc)
            db.commit()
    finally:
        db.close()


# ------------------------------------------------------------------ التطبيق
app = FastAPI(title="منصة رقمنة المخطوطات — نسخة مستقلة", version="1.0")

ALLOWED = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".pdf"}


@app.get("/")
def home():
    return FileResponse(BASE_DIR / "static" / "index.html")


@app.get("/api/status")
def system_status():
    return {
        "ocr_available": tesseract_available(),
        "ocr_hint": None if tesseract_available() else (
            "Tesseract غير مثبَّت — المنصة تعمل (رفع + تحسين صور) لكن دون تعرف "
            "ضوئي. للتثبيت: Windows: مثبّت UB-Mannheim مع اختيار اللغة العربية | "
            "macOS: brew install tesseract tesseract-lang | "
            "Linux: sudo apt install tesseract-ocr tesseract-ocr-ara"
        ),
    }


@app.post("/api/manuscripts")
async def upload(background: BackgroundTasks, file: UploadFile,
                 title: str = Form(...), call_number: str | None = Form(None)):
    ext = Path(file.filename or "").suffix.lower()
    if ext not in ALLOWED:
        raise HTTPException(400, f"صيغة غير مدعومة: {ext}")

    mid = str(uuid.uuid4())
    dest_dir = ARCHIVE_DIR / mid
    dest_dir.mkdir(parents=True)
    dest = dest_dir / (file.filename or f"upload{ext}")

    h = hashlib.sha256()
    with open(dest, "wb") as f:
        while chunk := await file.read(1 << 20):
            f.write(chunk)
            h.update(chunk)

    db = SessionLocal()
    try:
        db.add(Manuscript(id=mid, title=title, call_number=call_number,
                          original_path=str(dest), checksum_sha256=h.hexdigest()))
        db.commit()
    finally:
        db.close()

    # المعالجة في خيط منفصل حتى لا تُحجب الاستجابة (بديل Celery في نسخة بلا خدمات)
    threading.Thread(target=process_manuscript_job, args=(mid,), daemon=True).start()
    return {"id": mid, "status": "uploaded", "checksum_sha256": h.hexdigest()}


@app.get("/api/manuscripts")
def list_manuscripts():
    db = SessionLocal()
    try:
        rows = db.execute(select(Manuscript).order_by(Manuscript.uploaded_at.desc())).scalars().all()
        return [{"id": m.id, "title": m.title, "call_number": m.call_number,
                 "page_count": m.page_count, "status": m.status,
                 "error": m.error_message} for m in rows]
    finally:
        db.close()


@app.get("/api/manuscripts/{mid}/pages")
def list_pages(mid: str):
    db = SessionLocal()
    try:
        pages = db.execute(select(Page).where(Page.manuscript_id == mid)
                           .order_by(Page.page_number)).scalars().all()
        return [{"id": p.id, "page_number": p.page_number,
                 "avg_confidence": round(p.avg_confidence, 1),
                 "text_preview": p.full_text[:200]} for p in pages]
    finally:
        db.close()


@app.get("/api/pages/{pid}/text")
def page_text(pid: str):
    db = SessionLocal()
    try:
        p = db.get(Page, pid)
        if not p:
            raise HTTPException(404, "الصفحة غير موجودة")
        return {"page_number": p.page_number, "avg_confidence": p.avg_confidence,
                "full_text": p.full_text}
    finally:
        db.close()


@app.get("/api/pages/{pid}/image")
def page_image(pid: str, processed: bool = False):
    db = SessionLocal()
    try:
        p = db.get(Page, pid)
        if not p:
            raise HTTPException(404, "الصفحة غير موجودة")
        path = p.processed_image_path if processed and p.processed_image_path else p.original_image_path
        return FileResponse(path)
    finally:
        db.close()


@app.get("/api/pages/{pid}/crop")
def word_crop(pid: str, x: int, y: int, w: int, h: int):
    """قصاصة كلمة من الصورة الأصلية (المراجع يقرأ الحبر الأصلي أفضل من الثنائية)."""
    db = SessionLocal()
    try:
        p = db.get(Page, pid)
        if not p:
            raise HTTPException(404, "الصفحة غير موجودة")
        img = cv2.imread(p.original_image_path)
        if img is None:
            raise HTTPException(422, "تعذّرت قراءة الصورة")
        H, W = img.shape[:2]
        pad = 6
        crop = img[max(0, y - pad):min(H, y + h + pad), max(0, x - pad):min(W, x + w + pad)]
        ok, buf = cv2.imencode(".png", crop)
        return Response(content=buf.tobytes(), media_type="image/png")
    finally:
        db.close()


@app.get("/api/review/queue")
def review_queue(limit: int = 30):
    """الكلمات تحت العتبة، الأكثر غموضًا أولًا (uncertainty sampling)، دون تصحيح بعد."""
    db = SessionLocal()
    try:
        rows = db.execute(
            select(Word, Page, Manuscript)
            .join(Page, Word.page_id == Page.id)
            .join(Manuscript, Page.manuscript_id == Manuscript.id)
            .where(Word.confidence < REVIEW_THRESHOLD, Word.corrected_text.is_(None))
            .order_by(Word.confidence.asc()).limit(limit)
        ).all()
        return [{"word_id": w.id, "page_id": p.id, "manuscript_title": m.title,
                 "page_number": p.page_number, "predicted_text": w.text,
                 "confidence": round(w.confidence, 1),
                 "x": w.x, "y": w.y, "w": w.w, "h": w.h} for w, p, m in rows]
    finally:
        db.close()


@app.post("/api/review/correct")
def correct(word_id: str = Form(...), corrected_text: str = Form(...)):
    db = SessionLocal()
    try:
        w = db.get(Word, word_id)
        if not w:
            raise HTTPException(404, "الكلمة غير موجودة")
        w.corrected_text = corrected_text
        w.is_training_sample = True  # كل تصحيح = عيّنة تدريب مستقبلية تلقائيًا
        # تحديث نص الصفحة الكامل ليعكس التصحيح في البحث فورًا
        page = w.page
        if w.text and w.text in page.full_text:
            page.full_text = page.full_text.replace(w.text, corrected_text, 1)
            page.normalized_text = normalize_arabic(page.full_text)
        db.commit()
        remaining = db.execute(
            select(Word).where(Word.confidence < REVIEW_THRESHOLD,
                               Word.corrected_text.is_(None))
        ).scalars().all()
        return {"status": "saved", "remaining_in_queue": len(remaining),
                "training_samples_total": db.query(Word).filter(
                    Word.is_training_sample.is_(True)).count()}
    finally:
        db.close()


@app.get("/api/search")
def search(q: str, limit: int = 20):
    """بحث نصي بتطبيع عربي (يتجاهل التشكيل وفروق الهمزات/الياء/التاء المربوطة)."""
    nq = normalize_arabic(q.strip())
    if not nq:
        return []
    db = SessionLocal()
    try:
        rows = db.execute(
            select(Page, Manuscript).join(Manuscript)
            .where(Page.normalized_text.contains(nq)).limit(limit)
        ).all()
        results = []
        for p, m in rows:
            idx = p.normalized_text.find(nq)
            # القصاصة من النص الأصلي بموضع تقريبي مطابق
            start = max(0, idx - 60)
            snippet = p.full_text[start:start + 160].replace("\n", " ")
            results.append({"manuscript_title": m.title, "page_id": p.id,
                            "page_number": p.page_number, "snippet": f"...{snippet}..."})
        return results
    finally:
        db.close()


@app.get("/api/training/export")
def export_training_samples():
    """تصدير كل التصحيحات كبيانات تدريب (JSON) — جاهزة لضبط نموذج HTR لاحقًا."""
    db = SessionLocal()
    try:
        rows = db.execute(
            select(Word, Page).join(Page).where(Word.is_training_sample.is_(True))
        ).all()
        return [{"page_image": p.original_image_path, "bbox": [w.x, w.y, w.w, w.h],
                 "predicted": w.text, "ground_truth": w.corrected_text} for w, p in rows]
    finally:
        db.close()


app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")


if __name__ == "__main__":
    import uvicorn
    # السحابة (Render/Railway/Fly) تحقن المنفذ عبر متغير PORT وتتطلب host=0.0.0.0
    # محليًا تبقى القيم الافتراضية: 127.0.0.1:8000
    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", "8000"))
    print("=" * 60)
    print("  منصة رقمنة المخطوطات — النسخة المستقلة")
    print(f"  افتح المتصفح على:  http://localhost:{port}")
    if not tesseract_available():
        print("  ⚠ Tesseract غير مثبَّت: الرفع والتحسين يعملان، OCR معطَّل.")
        print("    راجع README.md لتعليمات التثبيت (دقيقتان).")
    print("=" * 60)
    uvicorn.run(app, host=host, port=port)
