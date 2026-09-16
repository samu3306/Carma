#!/usr/bin/env bash
set -euo pipefail

PROJECT_ID="${CARMA_PROJECT_ID:-$(gcloud config get-value project)}"
REGION="${CARMA_REGION:-asia-southeast1}"
SERVICE="${CARMA_SERVICE:-carma-web}"
INSTANCE="${CARMA_SQL_INSTANCE:-carma-postgres}"
DATABASE="${CARMA_DATABASE:-carma}"
DATABASE_USER="${CARMA_DATABASE_USER:-carma_app}"
BUCKET="${CARMA_MEDIA_BUCKET:-${PROJECT_ID}-carma-media}"
SECRET="${CARMA_DATABASE_SECRET:-carma-database-url}"

if [[ -z "${PROJECT_ID}" || "${PROJECT_ID}" == "(unset)" ]]; then
  echo "請先執行：gcloud config set project YOUR_PROJECT_ID" >&2
  exit 1
fi

PROJECT_NUMBER="$(gcloud projects describe "${PROJECT_ID}" --format='value(projectNumber)')"
RUNTIME_SERVICE_ACCOUNT="${CARMA_RUNTIME_SERVICE_ACCOUNT:-${PROJECT_NUMBER}-compute@developer.gserviceaccount.com}"

gcloud services enable \
  sqladmin.googleapis.com \
  secretmanager.googleapis.com \
  storage.googleapis.com \
  --project "${PROJECT_ID}"

if ! gcloud sql instances describe "${INSTANCE}" --project "${PROJECT_ID}" >/dev/null 2>&1; then
  gcloud sql instances create "${INSTANCE}" \
    --project "${PROJECT_ID}" \
    --database-version POSTGRES_15 \
    --tier db-f1-micro \
    --region "${REGION}" \
    --availability-type zonal \
    --storage-type SSD \
    --storage-size 10
fi

if ! gcloud sql databases describe "${DATABASE}" --instance "${INSTANCE}" --project "${PROJECT_ID}" >/dev/null 2>&1; then
  gcloud sql databases create "${DATABASE}" --instance "${INSTANCE}" --project "${PROJECT_ID}"
fi

if ! gcloud storage buckets describe "gs://${BUCKET}" --project "${PROJECT_ID}" >/dev/null 2>&1; then
  gcloud storage buckets create "gs://${BUCKET}" \
    --project "${PROJECT_ID}" \
    --location "${REGION}" \
    --uniform-bucket-level-access
fi

gcloud projects add-iam-policy-binding "${PROJECT_ID}" \
  --member "serviceAccount:${RUNTIME_SERVICE_ACCOUNT}" \
  --role roles/cloudsql.client >/dev/null
gcloud storage buckets add-iam-policy-binding "gs://${BUCKET}" \
  --member "serviceAccount:${RUNTIME_SERVICE_ACCOUNT}" \
  --role roles/storage.objectUser >/dev/null

INSTANCE_CONNECTION="$(gcloud sql instances describe "${INSTANCE}" --project "${PROJECT_ID}" --format='value(connectionName)')"
if ! gcloud secrets describe "${SECRET}" --project "${PROJECT_ID}" >/dev/null 2>&1; then
  DATABASE_PASSWORD="$(openssl rand -hex 24)"
  if gcloud sql users list --instance "${INSTANCE}" --project "${PROJECT_ID}" --format='value(name)' | grep -Fxq "${DATABASE_USER}"; then
    gcloud sql users set-password "${DATABASE_USER}" \
      --instance "${INSTANCE}" \
      --project "${PROJECT_ID}" \
      --password "${DATABASE_PASSWORD}"
  else
    gcloud sql users create "${DATABASE_USER}" \
      --instance "${INSTANCE}" \
      --project "${PROJECT_ID}" \
      --password "${DATABASE_PASSWORD}"
  fi
  DATABASE_URL="postgresql+psycopg://${DATABASE_USER}:${DATABASE_PASSWORD}@/${DATABASE}?host=/cloudsql/${INSTANCE_CONNECTION}"
  printf '%s' "${DATABASE_URL}" | gcloud secrets create "${SECRET}" --project "${PROJECT_ID}" --replication-policy automatic --data-file=-
fi
gcloud secrets add-iam-policy-binding "${SECRET}" \
  --project "${PROJECT_ID}" \
  --member "serviceAccount:${RUNTIME_SERVICE_ACCOUNT}" \
  --role roles/secretmanager.secretAccessor >/dev/null

gcloud run deploy "${SERVICE}" \
  --project "${PROJECT_ID}" \
  --region "${REGION}" \
  --source . \
  --allow-unauthenticated \
  --port 8080 \
  --cpu 2 \
  --memory 4Gi \
  --concurrency 4 \
  --min-instances 0 \
  --max-instances 1 \
  --timeout 300 \
  --service-account "${RUNTIME_SERVICE_ACCOUNT}" \
  --add-cloudsql-instances "${INSTANCE_CONNECTION}" \
  --set-secrets "CARMA_DATABASE_URL=${SECRET}:latest" \
  --set-env-vars "CARMA_UPLOAD_DIR=/mnt/carma/uploads,CARMA_ANALYSIS_DIR=/mnt/carma/uploads/analysis,CARMA_STORAGE_BACKEND=cloud_storage_mount,CARMA_SEED_ON_STARTUP=false" \
  --add-volume "mount-path=/mnt/carma,type=cloud-storage,bucket=${BUCKET},readonly=false"

SERVICE_URL="$(gcloud run services describe "${SERVICE}" --project "${PROJECT_ID}" --region "${REGION}" --format='value(status.url)')"
echo
echo "永久儲存部署完成：${SERVICE_URL}"
echo "確認狀態：${SERVICE_URL}/health"
