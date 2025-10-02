from fastapi import FastAPI
from app.routes import home, upload, predict, database
from google.cloud import storage
import os

app = FastAPI(title="StressXOR")

# Incluir routers (endpoints) definidos en cada módulo de routes
app.include_router(home.router)
app.include_router(upload.router)
app.include_router(predict.router)
app.include_router(database.router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

@app.get("/test_gcp")
async def test_gcp():
    try:
        blobs = storage_client.list_blobs("bucket-pfb_keepcoding2")
        blob_names = [blob.name for blob in blobs]
        return {"blobs": blob_names}
    except Exception as e:
        return {"error": str(e)}