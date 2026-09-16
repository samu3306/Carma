FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app
COPY requirements.txt .
RUN python -m pip install --no-cache-dir -r requirements.txt
COPY backend ./backend
RUN mkdir -p /data/uploads
ENV CARMA_DATABASE_URL=sqlite:////data/carma.db CARMA_UPLOAD_DIR=/data/uploads CARMA_ANALYSIS_DIR=/data/uploads/analysis
EXPOSE 8000
CMD ["sh", "-c", "python -m backend.app.seed && uvicorn backend.app.main:app --host 0.0.0.0 --port 8000"]
