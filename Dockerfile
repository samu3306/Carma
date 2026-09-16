FROM node:18-alpine AS frontend-build

WORKDIR /build
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend ./
RUN npm run build


FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8080 \
    CARMA_DATABASE_URL=sqlite:////tmp/carma/carma.db \
    CARMA_UPLOAD_DIR=/tmp/carma/uploads \
    CARMA_ANALYSIS_DIR=/tmp/carma/uploads/analysis \
    CARMA_DATASET_SOURCE_DIR=/tmp/carma/datasets \
    CARMA_FRONTEND_DIR=/app/frontend/dist \
    CARMA_QWEN_QUALITY_ENABLED=false \
    CARMA_QWEN_DAMAGE_ENABLED=false \
    CARMA_ANGLE_CLASSIFIER_ENABLED=true \
    CARMA_ANGLE_CLASSIFIER_BACKEND=dinov2_onnx \
    CARMA_DINOV2_DEVICE=cpu \
    CARMA_CLEANLINESS_ANALYSIS_ENABLED=false \
    CARMA_SEED_ON_STARTUP=true \
    CARMA_MODEL_VERSION=dinov2-onnx-int8-svm-cloud-v2

WORKDIR /app
COPY requirements.txt ./
RUN python -m pip install --no-cache-dir -r requirements.txt

COPY backend ./backend
COPY src ./src
COPY scripts/__init__.py scripts/train_retake_baseline.py ./scripts/
COPY models/yolov4-tiny.cfg models/yolov4-tiny.weights ./models/
COPY models/dinov2_angle_classifier/angle_svm.xml models/dinov2_angle_classifier/angle_metadata.npz models/dinov2_angle_classifier/dinov2_vits14_matmul_int8.onnx ./models/dinov2_angle_classifier/
COPY --from=frontend-build /build/dist ./frontend/dist

RUN mkdir -p /tmp/carma/uploads/analysis /tmp/carma/datasets
RUN python -c "import onnxruntime as ort; ort.InferenceSession('/app/models/dinov2_angle_classifier/dinov2_vits14_matmul_int8.onnx', providers=['CPUExecutionProvider']); print('DINOv2 ONNX validation passed')"

EXPOSE 8080
CMD ["sh", "-c", "if [ \"${CARMA_SEED_ON_STARTUP:-true}\" = \"true\" ]; then python -m backend.app.seed; fi && exec python -m uvicorn backend.app.main:app --host 0.0.0.0 --port ${PORT:-8080}"]
