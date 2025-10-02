from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from app.services.database import store_result_in_db_in_bucket
from app.config import BUCKET_NAME, DB_BLOB_PATH
from datetime import datetime

router = APIRouter()
templates = Jinja2Templates(directory="app/templates")

@router.get("/save_result", response_class=HTMLResponse)
def save_result(
    request: Request,
    video: str = Query(..., description="Ruta completa del video en el bucket"),
    model_type: str = Query(..., description="Tipo de modelo"),
    model_filename: str = Query(..., description="Ruta completa del modelo en el bucket"),
    avg_stress: float = Query(..., description="Media de estrés"),
    state_message: str = Query(..., description="Mensaje de estado")
):
    try:
        store_result_in_db_in_bucket(BUCKET_NAME, DB_BLOB_PATH, video, model_type, model_filename, avg_stress, state_message)
        return templates.TemplateResponse("save_database_confirmation.html", {"request": request})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))