import torch
import pickle
from fastapi import HTTPException
import xgboost as xgb
from catboost import CatBoostRegressor
import torch.nn as nn

#Clase de la NN·
class StressLSTM(nn.Module):
    def __init__(self, input_dim=133, hidden_dim=88, num_layers=2, dropout=0.4550597814102049):
        super(StressLSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.batch_norm = nn.BatchNorm1d(hidden_dim)  # Batch Normalization
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        lstm_out, _ = self.lstm(x.unsqueeze(1))  # (batch, seq_len=1, input_dim)
        lstm_out = self.batch_norm(lstm_out[:, -1, :])  # Última salida del LSTM con batch norm
        return self.fc(lstm_out)  # Pasar por la capa totalmente conectada
        
#Cargador de modelos#
def load_model(local_path: str, model_type: str):
    if model_type == "nn":
        try:
            model = StressLSTM()
            # Carga los datos guardados. Se admite que el archivo pueda tener la clave 'state_dict'
            loaded_data = torch.load(local_path, map_location=torch.device("cpu"))
            state_dict = loaded_data.get('state_dict', loaded_data)
            model.load_state_dict(state_dict)
            model.eval()
            return model
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error al cargar modelo NN: {str(e)}")
    elif model_type == "xgb":
        try:
            model = xgb.Booster()
            model.load_model(local_path)
            return model
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error al cargar modelo XGBoost: {str(e)}")
    elif model_type == "rf":
        try:
            with open(local_path, "rb") as f:
                model = pickle.load(f)
            return model
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error al cargar modelo RF: {str(e)}")
    elif model_type == "cbm":
        try:
            model = CatBoostRegressor()
            model.load_model(local_path)
            return model
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error al cargar modelo CBM: {str(e)}")
    else:
        raise HTTPException(status_code=400, detail="Tipo de modelo no reconocido")