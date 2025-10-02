import os
from google.cloud import storage
from app.config import BUCKET_NAME

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "app/credencial_GCP.json"
storage_client = storage.Client()

def download_blob_to_file(bucket_name: str, blob_name: str, destination_path: str):
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    os.makedirs(os.path.dirname(destination_path), exist_ok=True)
    blob.download_to_filename(destination_path)

def upload_file_to_bucket(bucket_name: str, file, folder: str) -> str:
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(f"{folder}/{file.filename}")
    blob.upload_from_file(file.file, content_type=file.content_type)
    return f"Archivo subido exitosamente a {folder}/{file.filename}"

def list_bucket_files(bucket_name: str, prefix: str = "") -> list:
    bucket = storage_client.bucket(bucket_name)
    return [blob.name for blob in bucket.list_blobs(prefix=prefix)]