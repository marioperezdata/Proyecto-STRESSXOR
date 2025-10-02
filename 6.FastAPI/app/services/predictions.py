import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import xgboost as xgb
from io import BytesIO
import base64
from fastapi import HTTPException
from sklearn.preprocessing import StandardScaler
from IPython.display import HTML

def predict_nn(features_df: pd.DataFrame, model) -> list:
    """Predicción para redes neuronales (ej. LSTM)."""
    try:
        data = features_df.drop(columns=['sec'])
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(data)
        scaled_tensor = torch.tensor(scaled_data, dtype=torch.float32)
        y = np.array(range(0,11)).reshape(-1, 1)
        scaled_y = scaler.fit_transform(y)
        # Padding si es necesario (ejemplo: se espera un tamaño fijo de 134 features)
        padding_size = 133 - scaled_tensor.shape[1]
        if padding_size > 0:
            padding = torch.zeros(scaled_tensor.shape[0], padding_size, dtype=torch.float32)
            scaled_tensor = torch.cat([scaled_tensor, padding], dim=1)
        with torch.no_grad():
            prediction = model(scaled_tensor)
        return [round(x) for x in scaler.inverse_transform(np.array(prediction.tolist()).reshape(-1, 1)).flatten()]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en predicción NN: {str(e)}")

def predict_xgb(features_df: pd.DataFrame, model) -> list:
    """Predicción para modelo XGBoost."""
    try:
        data = features_df.drop(columns=['sec'])
        # Ajuste de features: se asume que el modelo tiene un atributo feature_names
        feature_names_model = model.feature_names
        feature_names_input = data.columns.tolist()
        sobrantes = set(feature_names_input) - set(feature_names_model)
        faltantes = set(feature_names_model) - set(feature_names_input)
        data = data.drop(columns=list(sobrantes))
        for feature in faltantes:
            data[feature] = 0
        data = data[feature_names_model]
        dmatrix = xgb.DMatrix(data)
        preds = model.predict(dmatrix)
        return [round(p, 0) for p in preds.tolist()]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en predicción XGBoost: {str(e)}")

def predict_rf(features_df: pd.DataFrame, model) -> list:
    """Predicción para modelo Random Forest."""
    try:
        data = features_df.drop(columns=['sec'])
        feature_names_model = model.feature_names_in_
        feature_names_input = data.columns.tolist()
        sobrantes = set(feature_names_input) - set(feature_names_model)
        faltantes = set(feature_names_model) - set(feature_names_input)
        data = data.drop(columns=list(sobrantes))
        for feature in faltantes:
            data[feature] = 0
        data = data[feature_names_model]
        preds = model.predict(data)
        return preds.tolist()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en predicción RF: {str(e)}")
        
def predict_cbm(features_df: pd.DataFrame, model) -> dict:
    """Predicción para modelo CatBoost."""
    try:
        data = features_df.drop(columns=['sec'])
        feature_names_model = model.feature_names_
        feature_names_input = data.columns.tolist()
        sobrantes = set(feature_names_input) - set(feature_names_model)
        faltantes = set(feature_names_model) - set(feature_names_input)
        data = data.drop(columns=list(sobrantes))
        for feature in faltantes:
            data[feature] = 0
        data = data[feature_names_model]
        preds = model.predict(data)
        return preds.tolist()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en predicción CBM: {str(e)}")

def generate_stress_plot(predictions) -> str:
    """Genera un gráfico de estrés y lo devuelve como cadena base64."""
    plt.figure(figsize=(10, 5))
    plt.plot(predictions, label="Nivel de Estrés")
    plt.xlabel("Minuto")
    plt.ylabel("Valor de Estrés")
    plt.legend()
    buf = BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    img_base64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    plt.close()
    #html_code = f"<img src='data:image/png;base64,{img_base64}' />" 
    return img_base64#HTML(html_code)