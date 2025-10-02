# Proyecto STRESSXOR

📌 Descripción

Este proyecto aborda la predicción del nivel de estrés de personas durante tareas laborales, aplicando técnicas de Machine Learning y Deep Learning. Su propósito es proporcionar herramientas de apoyo para evaluar el comportamiento bajo presión, con aplicaciones en procesos de selección de personal y análisis de desempeño.

📊 Datos Utilizados

El estudio se basa en el dataset SWELL-KW, el cual recopila información multimodal de 25 participantes en escenarios de estrés inducido. Entre los datos registrados se incluyen:

Dataset A: Interacción con el ordenador (teclado, ratón, uso de software)

Dataset B: Expresiones faciales

Dataset C: Postura corporal

Dataset D: Señales biométricas (como ritmo cardíaco y actividad electrodermal)

# 📂 Estructura del Repositorio


```
📂 proyecto-prediccion-estres
│── 📜 README.md  # Este documento
│── 📜 Diagrama_Arquitectura.pdf  # Diagrama con enlaces a notebooks
│── 📜 requeriments.txt #  entorno de ejecución necesario
│── 📂 1.Analisis_data_set_y_validacion # (un notebook por dataset y el mergeado de los mismos
│── 📂 2.Preprocesado #de la data mergeada
│── 📂 3.Modelos  # Entrenamiento de los modelos propuestos
│── 📂 4.Extraccion_features_video  # extraccion de features y pruebas con los modelos 
│── 📂 5.MLFlow  #  Trackeo de modelos y registros de métricas e hiperparámetros en mlFlow
│── 📂 6.FastAPI  # Aplicación para predicción del estrés
│── 📂 7.PowerBI # Dashboard de resultados
│── 📂 8.Memoria_tecnica #Descripcion del proyecto completo
│── 📂 9.Presentacion_PPT
```


🏗️ Arquitectura del Proyecto

La estructura del proyecto ha sido documentada en el archivo "Diagrama_Arquitectura.pdf", donde se encuentran embebidos los enlaces a los Notebooks utilizados en cada fase del análisis. Se puede utilizar para un acceso más rápido y fácil seguimiento de los pasos realizados.
