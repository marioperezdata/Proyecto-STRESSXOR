import cv2
import pandas as pd
from app.services.feature_extraction_helpers import extract_all_features, face_mesh, pose


def process_video(video_path: str) -> pd.DataFrame:
    """
        Procesa el video en la ruta dada y extrae las features de cada frame,
        para luego agruparlas por segundo (calculando el promedio de cada feature).
        """
    cap = cv2.VideoCapture(video_path)
    FPS = cap.get(cv2.CAP_PROP_FPS)
    all_frames_features = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # Extrae las features usando la función definida previamente
        features = extract_all_features(frame, face_mesh, pose)
        all_frames_features.append(features)
    
    cap.release()
    
    # Convierte la lista de features en un DataFrame
    features_df = pd.DataFrame(all_frames_features)
    
    # Agrupa las features por segundo calculando el promedio
    second_data = []
    num_frames = len(features_df)
    frames_per_sec = int(FPS * 1) if FPS else 1
    num_sec = (num_frames + frames_per_sec - 1) // frames_per_sec
    
    for sec in range(num_sec):
        start = sec * frames_per_sec
        end = start + frames_per_sec
        sec_df = features_df.iloc[start:end]
        sec_mean = sec_df.mean().to_dict()
        sec_mean['sec'] = sec + 1
        second_data.append(sec_mean)
    
    return pd.DataFrame(second_data)

#import cv2
#import pandas as pd
#import gc
#from pyspark.sql import SparkSession, Row
#from pyspark.sql.functions import floor, col, avg
#from app.services.feature_extraction_helpers import extract_all_features_local, face_mesh, pose
#
#def init_spark():
#    # Inicializa Spark usando 2 núcleos (ajusta según tu entorno)
#    return SparkSession.builder.master("local[2]").appName("StressXOR_VideoProcessing").getOrCreate()
#def extract_frame_features_spark(frame):
#
#    # Utiliza la versión local que crea sus propias instancias de face_mesh y pose.
#    return extract_all_features_local(frame)
#def process_video(video_path: str, batch_size: int = 10) -> pd.DataFrame:
#    """
#    Procesa el video en lotes pequeños para extraer las características de cada frame utilizando PySpark.
#    Se evita cargar todos los frames en memoria a la vez. Los resultados se agrupan por segundo y se devuelven
#    como un DataFrame de Pandas.
#    """
#    # Iniciar Spark
#    spark = init_spark()
#
#    # Abrir el video y obtener FPS
#    cap = cv2.VideoCapture(video_path)
#    FPS = cap.get(cv2.CAP_PROP_FPS)
#    
#    all_features_list = []  # Aquí se acumularán las features procesadas de cada frame
#    frame_index = 0
#
#    while True:
#        batch_frames = []
#        # Leer un batch de frames
#        for _ in range(batch_size):
#            ret, frame = cap.read()
#            if not ret:
#                break
#            batch_frames.append(frame)
#        
#        if not batch_frames:
#            break  # Salir si no se leyeron más frames
#        
#        # Crear un RDD para este batch
#        batch_rdd = spark.sparkContext.parallelize(batch_frames)
#        # Procesar los frames en paralelo en el batch
#        batch_features = batch_rdd.map(lambda frame: extract_all_features_local(frame, face_mesh, pose)).collect()
#        
#        # Asignar un índice a cada frame procesado para poder agrupar por segundo
#        for features in batch_features:
#            features["frame_index"] = frame_index
#            frame_index += 1
#            all_features_list.append(features)
#        
#        # Liberar la memoria de este batch
#        del batch_frames, batch_features
#        gc.collect()
#
#    cap.release()
#    
#    # Convertir la lista de features a un DataFrame de Pandas
#    features_df = pd.DataFrame(all_features_list)
#    
#    # Agrupar las features por segundo
#    frames_per_sec = int(FPS) if FPS else 1
#    num_frames = len(features_df)
#    num_sec = (num_frames + frames_per_sec - 1) // frames_per_sec
#    
#    second_data = []
#    for sec in range(num_sec):
#        start = sec * frames_per_sec
#        end = start + frames_per_sec
#        sec_df = features_df.iloc[start:end]
#        sec_mean = sec_df.mean().to_dict()
#        sec_mean['sec'] = sec + 1
#        second_data.append(sec_mean)
#    
#    # Detener la sesión de Spark para liberar recursos
#    spark.stop()
#    del all_features_list
#    gc.collect()
#    
#    return pd.DataFrame(second_data)