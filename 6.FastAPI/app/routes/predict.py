from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from app.services.gcp_storage import download_blob_to_file, list_bucket_files
from app.services.model_loader import load_model
from app.services.feature_extraction import process_video
from app.services.predictions import predict_nn, predict_xgb, predict_rf, predict_cbm, generate_stress_plot
from app.services.database import store_result_in_db_in_bucket
from app.config import BUCKET_NAME, DB_BLOB_PATH
import os

router = APIRouter()
templates = Jinja2Templates(directory="app/templates")

@router.get("/predict_page", response_class=HTMLResponse)
def predict_page(request: Request):
    """Página con formulario para predecir estrés, con botón estilizado en verde."""
    # Listar archivos disponibles en el bucket
    model_files = list_bucket_files(BUCKET_NAME, prefix="Modelos/")
    video_files = list_bucket_files(BUCKET_NAME, prefix="Videos/")
    model_options = "".join([f'<option value="{f}">{f.split("/")[-1]}</option>' for f in model_files])
    video_options = "".join([f'<option value="{f}">{f.split("/")[-1]}</option>' for f in video_files])
    
    return templates.TemplateResponse("predict_page.html", {
        "request": request,
        "model_options": model_options,
        "video_options": video_options
    })

@router.get("/predict", response_class=HTMLResponse)
def predict_stress(
    request: Request,
    video: str = Query(..., description="Ruta completa del video en el bucket (ej. Videos/video.mp4)"),
    model_type: str = Query(..., description="Tipo de modelo: nn, xgb, rf"),
    model_filename: str = Query(..., description="Ruta completa del modelo en el bucket (ej. Modelos/modelo_LSTM.pkl)")
):
    try:
      #Comrpobar tipo de modelo correcto
        filename_lower = model_filename.lower()
        if model_type == "nn":
            if not filename_lower.endswith('.pth'):
                raise HTTPException(status_code=400, detail="El modelo no es del tipo nn")
        elif model_type == "xgb":
            if not filename_lower.endswith('.ubj'):
                raise HTTPException(status_code=400, detail="El modelo no es del tipo xgboost")
        elif model_type == "rf":
            if not filename_lower.endswith('.pkl'):
                raise HTTPException(status_code=400, detail="El modelo no es del tipo rf")
        elif model_type == "cbm":
            if not filename_lower.endswith('.cbm'):
                raise HTTPException(status_code=400, detail="El modelo no es del tipo cbm")
        else:
            raise HTTPException(status_code=400, detail="Tipo de modelo no soportado")
    except Exception as e:
        error_message = str(e)
        return templates.TemplateResponse("predict_error.html", {"request": request, "error_message": error_message}, status_code=500)
    try:
        # Descarga el video
        video_temp_path = f"/tmp/{video.split('/')[-1]}"
        download_blob_to_file(BUCKET_NAME, video, video_temp_path)

        #Comrpobar tipo de modelo correcto
        filename_lower = model_filename.lower()
        if model_type == "nn":
            if not filename_lower.endswith('.pth'):
                raise HTTPException(status_code=400, detail="El modelo no es del tipo nn")
        elif model_type == "xgb":
            if not filename_lower.endswith('.ubj'):
                raise HTTPException(status_code=400, detail="El modelo no es del tipo xgboost")
        elif model_type == "rf":
            if not filename_lower.endswith('.pkl'):
                raise HTTPException(status_code=400, detail="El modelo no es del tipo rf")
        elif model_type == "cbm":
            if not filename_lower.endswith('.cbm'):
                raise HTTPException(status_code=400, detail="El modelo no es del tipo cbm")
        else:
            raise HTTPException(status_code=400, detail="Tipo de modelo no soportado")
        
        # Procesa el video y extrae las features
        sec_features_df = process_video(video_temp_path)
        
        # Descarga y carga el modelo desde la carpeta Modelos
        model_temp_path = f"/tmp/{model_filename.split('/')[-1]}"
        download_blob_to_file(BUCKET_NAME, model_filename, model_temp_path)

        # Carga el modelo desde el bucket
        model = load_model(model_temp_path, model_type)
        
        if model_type == "nn":
            pred_value = predict_nn(sec_features_df, model)
            predictions = [pred_value]
        elif model_type == "xgb":
            pred_value = predict_xgb(sec_features_df, model)
            predictions = [pred_value]
        elif model_type == "rf":
            pred_value = predict_rf(sec_features_df, model)
            predictions = [pred_value]
        elif model_type == "cbm" :
            pred_value = predict_cbm(sec_features_df, model)
            predictions = [pred_value]
        else:
            raise HTTPException(status_code=400, detail="Tipo de modelo no soportado")
        
        # Calcula la media y formatea las predicciones
        flat_predictions = [item for sublist in predictions for item in (sublist if isinstance(sublist, list) else [sublist])]

        avg_stress = sum(flat_predictions) / len(flat_predictions) if flat_predictions else 0
        predictions_str = ", ".join(str(round(val, 1)) for val in flat_predictions)
        
        # Define estilos según la media de estrés
        if avg_stress <= 2:
            box_color = "#0000FF"      # Azul
            bg_color = "#ADD8E6"       # Azul claro
            state_message = "Relajado"
        elif avg_stress <= 4:
            box_color = "#008000"      # Verde
            bg_color = "#90EE90"       # Verde claro
            state_message = "Estrés bajo"
        elif avg_stress <= 7:
            box_color = "#FFA500"      # Naranja
            bg_color = "#FFDAB9"       # Naranja claro
            state_message = "Estrés moderado"
        else:
            box_color = "#FF0000"      # Rojo
            bg_color = "#FFC0CB"       # Rojo claro
            state_message = "Estrés alto"

        # Genera el gráfico de estrés en base64
        stress_graph = generate_stress_plot(flat_predictions)

        return templates.TemplateResponse("predict_results.html", {
            "request": request,
            "model_used": model_filename.split("/")[-1],
            "avg_stress": round(avg_stress, 1),
            "state_message": state_message,
            "stress_graph": stress_graph,
            "video": video,
            "model_type": model_type,
            "model_filename": model_filename,
            "box_color": box_color,  # Asegúrate de pasar estos valores
            "bg_color": bg_color
        })
    except Exception as e:
        error_message = str(e)
        return templates.TemplateResponse("predict_error.html", {"request": request, "error_message": error_message}, status_code=500)