from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

router = APIRouter()
templates = Jinja2Templates(directory="app/templates")

@router.get("/", response_class=HTMLResponse)
def home(request: Request):
    """Página principal: opciones para subir archivos o predecir estrés."""
    return templates.TemplateResponse("home.html", {"request": request})