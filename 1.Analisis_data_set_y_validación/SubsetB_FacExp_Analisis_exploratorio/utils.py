import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import math
import scipy.stats as stats
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import PowerTransformer, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import FunctionTransformer

#UTILS SUBSET A ANALYSIS

def cargar_datos_csv(ruta_archivo):
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
            sep=";",                # Separador de columnas
            decimal=",",            # Punto como separador decimal
            skip_blank_lines=True,  # Evita filas vacías
            na_values=[";;"],       # Trata ",," como valores nulos
            engine="python"         # Usa el motor de Python para mayor flexibilidad
        )

        # Normalizar espacios en los nombres de columnas
        df.columns = df.columns.str.strip()

        # Convertir timestamp a datetime
        if 'timestamp' in df.columns:
          df['timestamp'] = df['timestamp'].astype(str).str.strip() # Limpia espacios
          df['timestamp'] = pd.to_datetime(df['timestamp'], format="%Y%m%dT%H%M%S%f", errors='coerce')

        print(f"Datos cargados correctamente: {ruta_archivo}")
        return df

    except Exception as e:
        print(f"Error al cargar el archivo {ruta_archivo}: {e}")
        return None

def eliminar_columnas(df, columnas_a_eliminar):
    """
    Elimina columnas específicas de un DataFrame.

    Parámetros:
    df (pd.DataFrame): DataFrame original.
    columnas_a_eliminar (list): Lista con los nombres de las columnas a eliminar.

    Retorna:
    pd.DataFrame: DataFrame sin las columnas especificadas.
    """
    columnas_presentes = [col for col in columnas_a_eliminar if col in df.columns]
    df = df.drop(columns=columnas_presentes, errors='ignore')
    print(f"Se eliminaron las columnas: {columnas_presentes}")
    return df

def comparar_columnas(df1, df2):
    """
    Compara los nombres de las columnas entre dos dataframes.

    Parámetros:
    df1 (str): dataframe 1
    df2 (str): dataf6rame 2
    """
    
    columnas_1 = set(df1.columns)
    columnas_2 = set(df2.columns)

    print("Columnas en común:", columnas_1 & columnas_2)
    print("Columnas solo en el primer archivo:", columnas_1 - columnas_2)
    print("Columnas solo en el segundo archivo:", columnas_2 - columnas_1)

def evaluar_datos_faltantes(df):
    """
    Evalúa los datos faltantes y los representa en una tabla clara.

    Parámetros:
    df (pd.DataFrame): DataFrame a analizar.

    Retorna:
    pd.DataFrame: Tabla con la evaluación de datos faltantes.
    """
    missing_data = df.isnull().sum()
    missing_percent = (missing_data / len(df)) * 100

    tabla_nulos = pd.DataFrame({
        'Columna': df.columns,
        'Valores Faltantes': missing_data,
        'Porcentaje (%)': missing_percent
    }).sort_values(by='Porcentaje (%)', ascending=False)

    return tabla_nulos

#EVALUCIÓN DE COLUMNAS DE UN MISMO ARCHIVO

def calcular_correlacion(df, col1, col2):
    """
    Calcula la correlación entre dos columnas de un DataFrame.

    Parámetros:
    df (pd.DataFrame): DataFrame con los datos.
    col1 (str): Nombre de la primera columna.
    col2 (str): Nombre de la segunda columna.

    Retorna:
    float: Valor de la correlación.
    """
    if col1 in df.columns and col2 in df.columns:
        correlacion = df[col1].corr(df[col2])
        print(f"Correlación entre {col1} y {col2}: {correlacion:.4f}")
        return correlacion
    else:
        print(f"Una de las columnas {col1} o {col2} no existe en el DataFrame.")
        return None

def graficar_histogramas_2col(df, col1, col2):
    """
    Genera histogramas comparativos de dos columnas.

    Parámetros:
    df (pd.DataFrame): DataFrame con los datos.
    col1 (str): Nombre de la primera columna.
    col2 (str): Nombre de la segunda columna.
    """
    if col1 in df.columns and col2 in df.columns:
        plt.figure(figsize=(10, 5))
        sns.histplot(df[col1], bins=30, kde=True, label=col1, color="blue", alpha=0.5)
        sns.histplot(df[col2], bins=30, kde=True, label=col2, color="red", alpha=0.5)
        plt.legend()
        plt.title(f"Distribución de {col1} vs {col2}")
        plt.xlabel("Distancia del mouse")
        plt.ylabel("Frecuencia")
        plt.show()
    else:
        print(f"Una de las columnas {col1} o {col2} no existe en el DataFrame.")

def graficar_scatter(df, col1, col2):
    """
    Genera un scatter plot para comparar dos columnas.

    Parámetros:
    df (pd.DataFrame): DataFrame con los datos.
    col1 (str): Nombre de la primera columna.
    col2 (str): Nombre de la segunda columna.
    """
    if col1 in df.columns and col2 in df.columns:
        plt.figure(figsize=(6, 6))
        sns.scatterplot(x=df[col1], y=df[col2], alpha=0.5)
        plt.title(f"Comparación {col1} vs {col2}")
        plt.xlabel(col1)
        plt.ylabel(col2)
        plt.show()
    else:
        print(f"Una de las columnas {col1} o {col2} no existe en el DataFrame.")


#ANALISIS EXPLORATOTIO

def graficar_boxplots(df, columnas_numericas):
    """
    Genera boxplots de las features por cada Condition en una única figura.

    Parámetros:
    df (pd.DataFrame): DataFrame con los datos.
    columnas_numericas (list): Lista de columnas numéricas a graficar.
    """
    if "Condition" not in df.columns:
        print("La columna 'Condition' no existe en el DataFrame.")
        return

    # Definir la cantidad de filas y columnas en la cuadrícula
    num_vars = len(columnas_numericas)
    cols = 4  # Número de columnas en la cuadrícula
    rows = math.ceil(num_vars / cols)  # Número de filas, ajustado dinámicamente

    # Crear la figura y los ejes
    fig, axes = plt.subplots(rows, cols, figsize=(20, 5 * rows))  # Ajusta tamaño
    axes = axes.flatten()  # Asegurar acceso a los ejes como una lista

    # Graficar cada boxplot en su correspondiente subplot
    for i, col in enumerate(columnas_numericas):
        if col in df.columns:
            sns.boxplot(x="Condition", y=col, data=df, ax=axes[i])
            axes[i].set_title(f"Distribución de {col} por Condition")
            axes[i].tick_params(axis="x", rotation=45)
        else:
            axes[i].axis("off")  # Ocultar subplot vacío si hay menos columnas de las esperadas
            print(f"La columna '{col}' no existe en el DataFrame.")

    # Ajustar espaciado
    plt.tight_layout()
    plt.show()

def correlacion_por_condition(df, columnas_numericas):
    """
    Genera matrices de correlación dentro de cada 'Condition', con un heatmap estilizado.

    Parámetros:
    df (pd.DataFrame): DataFrame con los datos.
    columnas_numericas (list): Lista de columnas numéricas a analizar.
    """
    if "Condition" not in df.columns:
        print("La columna 'Condition' no existe en el DataFrame.")
        return

    condiciones = df["Condition"].unique()

    for condition in condiciones:
        subset = df[df["Condition"] == condition]
        corr_matrix = np.abs(subset[columnas_numericas].corr())

        # Crear máscara para la mitad superior
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

        # Configurar figura
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, mask=mask, vmin=0.0, vmax=1.0, center=0.5,
                    linewidths=0.5, cmap="YlGnBu", cbar_kws={"shrink": .8},
                    annot=True, fmt=".2f")

        plt.title(f"Matriz de Correlación - Condition: {condition}")
        plt.show()

def graficar_histogramas(df, bins=50, figsize=(20, 10)):
    """
    Genera histogramas para todas las columnas numéricas de un DataFrame.

    Parámetros:
    df (pd.DataFrame): DataFrame con los datos.
    bins (int): Número de bins (intervalos) para los histogramas.
    figsize (tuple): Tamaño de la figura (ancho, alto).
    """
    df.hist(bins=bins, figsize=figsize)
    plt.tight_layout()
    plt.show()


def graficar_histogramas_por_condition(df, bins=50, figsize=(20, 10)):
    """
    Genera histogramas de las columnas numéricas, separados por cada Condition.

    Parámetros:
    df (pd.DataFrame): DataFrame con los datos.
    bins (int): Número de bins (intervalos) para los histogramas.
    figsize (tuple): Tamaño de la figura (ancho, alto).
    """
    if "Condition" not in df.columns:
        print("La columna 'Condition' no existe en el DataFrame.")
        return

    condiciones = df["Condition"].unique()
    
    for condition in condiciones:
        subset = df[df["Condition"] == condition]
        subset.hist(bins=bins, figsize=figsize)
        plt.suptitle(f"Histogramas para Condition: {condition}")
        plt.tight_layout()
        plt.show()

def comparar_histogramas(df1, df2, bins=50):
    """
    Genera histogramas comparativos de las variables numéricas en sheet_1 y sheet_2.

    Parámetros:
    df1 (pd.DataFrame): Primer DataFrame.
    df2 (pd.DataFrame): Segundo DataFrame.
    columnas_numericas (list): Lista de columnas numéricas a graficar.
    bins (int): Número de bins para los histogramas.
    """
    # Identificar columnas numéricas presentes en ambos DataFrames
    columnas_numericas_1 = set(df1.select_dtypes(include=['number']).columns)
    columnas_numericas_2 = set(df2.select_dtypes(include=['number']).columns)
    columnas_numericas = list(columnas_numericas_1.intersection(columnas_numericas_2))

    if not columnas_numericas:
        print("No hay columnas numéricas en común entre los DataFrames.")
    num_vars = len(columnas_numericas)
    cols = 3  # Número de columnas en la cuadrícula
    rows = math.ceil(num_vars / cols)  # Número de filas, ajustado dinámicamente

    fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))  # Ajustar tamaño
    axes = axes.flatten()  # Convertir la cuadrícula en una lista de ejes

    for i, col in enumerate(columnas_numericas):
        if col in df1.columns and col in df2.columns:
            sns.histplot(df1[col], bins=bins, kde=True, alpha=0.5, label="Sheet 1", ax=axes[i], color="blue")
            sns.histplot(df2[col], bins=bins, kde=True, alpha=0.5, label="Sheet 2", ax=axes[i], color="orange")
            axes[i].set_title(f"{col}")
            axes[i].legend()
        else:
            axes[i].axis("off")  # Oculta subplots vacíos si hay menos columnas de las esperadas
            print(f"La columna '{col}' no existe en ambos DataFrames.")

    plt.tight_layout()  # Ajustar espaciado
    plt.show()

def graficar_barras_estado_animo(sheet_1, sheet_2, figsize=(20, 10)):
    """
    Genera gráficos de barras comparando los estados de ánimo entre PP para los DataFrames sheet_1 y sheet_2.

    Parámetros:
    sheet_1 (pd.DataFrame): Primer DataFrame con los datos.
    sheet_2 (pd.DataFrame): Segundo DataFrame con los datos.
    figsize (tuple): Tamaño de la figura (ancho, alto).
    """
    # Lista de columnas de estados de ánimo
    columnas_estados = [
        "Sneutral", "Shappy", "Ssad", "Sangry", 
        "Ssurprised", "Sscared", "Sdisgusted"
    ]

    # Verificar que las columnas 'PP' y las de emociones estén presentes en ambos DataFrames
    if "PP" not in sheet_1.columns or "PP" not in sheet_2.columns or not all(col in sheet_1.columns for col in columnas_estados) or not all(col in sheet_2.columns for col in columnas_estados):
        print("Las columnas 'PP' o las de los estados de ánimo no existen en uno o ambos DataFrames.")
        return

    # Agrupar por PP y calcular el promedio de los estados de ánimo en ambos DataFrames
    sheet_1_avg = sheet_1.groupby("PP")[columnas_estados].mean()
    sheet_2_avg = sheet_2.groupby("PP")[columnas_estados].mean()

    # Iterar sobre cada valor de PP
    for pp_value in sheet_1["PP"].unique():
        if pp_value in sheet_1_avg.index and pp_value in sheet_2_avg.index:
            # Extraer los valores de los estados de ánimo para este PP
            sheet_1_vals = sheet_1_avg.loc[pp_value]
            sheet_2_vals = sheet_2_avg.loc[pp_value]

            # Crear el gráfico de barras para los estados de ánimo
            plt.figure(figsize=figsize)
            bar_width = 0.35
            index = range(len(columnas_estados))

            # Barras para sheet_1
            plt.bar(index, sheet_1_vals, bar_width, label=f"sheet_1 - PP: {pp_value}", alpha=0.7)

            # Barras para sheet_2
            plt.bar([i + bar_width for i in index], sheet_2_vals, bar_width, label=f"sheet_2 - PP: {pp_value}", alpha=0.7)

            # Personalización del gráfico
            plt.xlabel("Estado de ánimo")
            plt.ylabel("Promedio")
            plt.title(f"Comparación de estados de ánimo para PP: {pp_value}")
            plt.xticks([i + bar_width / 2 for i in index], columnas_estados, rotation=45)
            plt.legend()
            plt.tight_layout()
            plt.show()
        else:
            print(f"PP: {pp_value} no está presente en ambos DataFrames.")
#FUNCIONES DE PREPROCESAMIENTO

def auto_data_transform(df, columnas=None):
    """
    Detecta la distribución de cada columna en un DataFrame y aplica la mejor transformación.

    Parámetros:
    df (pd.DataFrame): Dataset de entrada.
    columnas (list): Lista de columnas a analizar (opcional). Si es None, analiza todas las numéricas.

    Retorna:
    df_transformado (pd.DataFrame): Dataset con las columnas transformadas.
    distribuciones (dict): Diccionario con la distribución detectada y la transformación aplicada.
    """
    if columnas is None:
        columnas = df.select_dtypes(include=['number']).columns

    df_transformado = df.copy()
    distribuciones = {}

    for col in columnas:
        data = df[col].dropna()  # Eliminar NaN para análisis

        if data.empty:
            continue  # Saltar columnas vacías

        # Prueba de normalidad de Shapiro-Wilk (p < 0.05 significa que no es normal)
        stat, p_value = stats.shapiro(data)
        es_normal = p_value > 0.05

        if es_normal:
            distribuciones[col] = ('Normal', 'Sin transformación')
            continue  # Si es normal, no aplicamos transformación

        # Detectar si es exponencial o altamente sesgada
        skewness = stats.skew(data)

        if skewness > 1:
            # Solo aplicar log si todos los valores son positivos
            if (df[col] > 0).all():
                df_transformado[col] = np.log1p(df[col] + 1e-6)  # Evitar log(0)
                distribuciones[col] = ('Exponencial/Sesgada', 'Logarítmica')
            else:
                # Aplicar Yeo-Johnson si hay valores negativos
                pt = PowerTransformer(method='yeo-johnson')
                df_transformado[col] = pt.fit_transform(df[[col]])
                distribuciones[col] = ('Exponencial/Sesgada', 'Yeo-Johnson')

        elif 0.5 < skewness <= 1:
            # Intentamos transformación raíz cuadrada
            df_transformado[col] = np.sqrt(df[col] - df[col].min() + 1)  # Evita sqrt(negativos)
            distribuciones[col] = ('Moderadamente sesgada', 'Raíz cuadrada')

        else:
            # Si no encaja en lo anterior, aplicamos Box-Cox si todos los valores son positivos
            if (df[col] > 0).all():
                pt = PowerTransformer(method='box-cox')
                df_transformado[col] = pt.fit_transform(df[[col]])
                distribuciones[col] = ('Desconocida', 'Box-Cox')
            else:
                # Para datos con negativos, usamos Yeo-Johnson
                pt = PowerTransformer(method='yeo-johnson')
                df_transformado[col] = pt.fit_transform(df[[col]])
                distribuciones[col] = ('Desconocida', 'Yeo-Johnson')

    return df_transformado, distribuciones

def load_and_clean_data(csv_path, numeric_features):
    """
    Carga el dataset, convierte columnas numéricas y reemplaza 999.0 por NaN.

    Parámetros:
    -----------
    csv_path : str
        Ruta del archivo CSV.

    Retorna:
    --------
    df : pd.DataFrame
        DataFrame limpio con datos convertidos.
    """
    df = pd.read_csv(csv_path)

    # Convertir solo las columnas numéricas
    for column in numeric_features:
        df[column] = pd.to_numeric(df[column], errors='coerce')

    # Reemplazar 999.0 con NaN en columnas numéricas
    df[numeric_features] = df[numeric_features].replace(999.0, np.nan)

    return df

# Crear pipeline de preprocesamiento para variables numéricas
def create_numeric_pipeline():
    """
    Crea el pipeline de preprocesamiento para variables numéricas.

    Retorna:
    --------
    numeric_pipeline : sklearn.pipeline.Pipeline
        Pipeline de preprocesamiento para datos numéricos.
    """
    numeric_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('yeo', PowerTransformer(method='yeo-johnson'))
    ])
    
    return numeric_pipeline

# Crear pipeline de preprocesamiento para variables categóricas
def create_categorical_pipeline():
    """
    Crea el pipeline de preprocesamiento para variables categóricas.

    Retorna:
    --------
    categorical_pipeline : sklearn.pipeline.Pipeline
        Pipeline de preprocesamiento para datos categóricos.
    """
    categorical_pipeline = Pipeline([
        ('encoder', OneHotEncoder(drop='first', sparse_output=False))
    ])
    
    return categorical_pipeline

# Aplicar preprocesamiento con ColumnTransformer
def apply_preprocessing(df, numeric_features, categorical_features):
    """
    Aplica el preprocesamiento a los datos usando ColumnTransformer.

    Parámetros:
    -----------
    df : pd.DataFrame
        DataFrame con los datos originales.
    numeric_features : list
        Lista de nombres de las columnas numéricas.
    categorical_features : list
        Lista de nombres de las columnas categóricas.

    Retorna:
    --------
    df_processed : pd.DataFrame
        DataFrame con los datos preprocesados.
    """
    # Crear el ColumnTransformer
    preprocessing = ColumnTransformer(
        transformers=[
            ('num', create_numeric_pipeline(), numeric_features),
            ('cat', create_categorical_pipeline(), categorical_features)
        ]
    )

    # Aplicar transformación
    P_preprocessed = preprocessing.fit_transform(df)

    # Obtener nombres de columnas
    columns_names = numeric_features + list(preprocessing.named_transformers_['cat'].get_feature_names_out(categorical_features))

    # Convertir a DataFrame
    df_processed = pd.DataFrame(P_preprocessed, columns=columns_names)

    return df_processed

# Función principal para preprocesar los datos de interacción con el ordenador
def preprocess_data(csv_path,numeric_features,categorical_features):
    """
    Función principal que preprocesa los datos de interacción con el ordenador.

    Parámetros:
    -----------
    csv_path : str
        Ruta del archivo CSV.

    Retorna:
    --------
    df_processed : pd.DataFrame
        DataFrame con los datos preprocesados.
    """
    df = load_and_clean_data(csv_path, numeric_features)
    df_processed = apply_preprocessing(df, numeric_features, categorical_features)
    
    return df_processed