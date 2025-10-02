import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import math
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import PowerTransformer, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import FunctionTransformer

#UTILS SUBSET A ANALYSIS

def cargar_datos_csv(ruta_archivo,sep=','):
    """
    Carga un archivo CSV y convierte la columna 'timestamp' a datetime.

    Parámetros:
    ruta_archivo (str): Ruta del archivo CSV.

    Retorna:
    pd.DataFrame: DataFrame con los datos cargados y 'timestamp' en formato datetime.
    """
    try:
        df = pd.read_csv(
            ruta_archivo,
            sep=sep,                # Separador de columnas
            decimal=".",            # Punto como separador decimal
            skip_blank_lines=True,  # Evita filas vacías
            na_values=[",,"],       # Trata ",," como valores nulos
            engine="python"         # Usa el motor de Python para mayor flexibilidad
        )

        # Normalizar espacios en los nombres de columnas
        df.columns = df.columns.str.strip()

        # Convertir timestamp a datetime
        if 'timestamp' in df.columns:
            df['timestamp'] = df['timestamp'].astype(str).str.strip()  # Limpia espacios
            df['timestamp'] = pd.to_datetime(df['timestamp'], format="%Y%m%dT%H%M%S%f", errors='coerce')

        print(f"Datos cargados correctamente: {ruta_archivo}")
        return df

    except Exception as e:
        print(f"Error al cargar el archivo {ruta_archivo}: {e}")
        return None
