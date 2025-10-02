from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from app.services.gcp_storage import upload_file_to_bucket
from app.config import BUCKET_NAME

router = APIRouter()
templates = Jinja2Templates(directory="app/templates")

@router.get("/upload_page", response_class=HTMLResponse)
def upload_page(request: Request):
    
    return templates.TemplateResponse("upload_page.html", {"request": request})

@router.post("/upload", response_class=HTMLResponse)
async def upload(request: Request, file: UploadFile = File(...), file_type: str = Form(...)):
    # Validación por extensión
    allowed_video_extensions = {".mp4", ".avi", ".mov"}
    allowed_model_extensions = {".pth", ".pkl", ".ubj", ".cbm"}
    filename = file.filename.lower()
    try:
      if file_type == "Videos":
        if not any(filename.endswith(ext) for ext in allowed_video_extensions):
            raise HTTPException(status_code=400, detail="El archivo no es un video válido.")
      elif file_type == "Modelos":
        if not any(filename.endswith(ext) for ext in allowed_model_extensions):
            raise HTTPException(status_code=400, detail="El archivo no es un modelo válido.")
      else:
        raise HTTPException(status_code=400, detail="Tipo de archivo no válido")
    except Exception as e:
        error_message = str(e)
        return templates.TemplateResponse("predict_error.html", {"request": request, "error_message": error_message}, status_code=500)
    try:
        result = upload_file_to_bucket(BUCKET_NAME, file, file_type)
        return templates.TemplateResponse("upload_confirmation.html", {"request": request, "message": result})
    except Exception as e:
        error_message = str(e)
        return templates.TemplateResponse("predict_error.html", {"request": request, "error_message": error_message}, status_code=500)