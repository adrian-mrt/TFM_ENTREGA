import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from transformers import CleanTransformer, EncodeTransformer, ImputeTransformer, LabelEncodingTransformer, PCAToDataFrameTransformer

def process_and_add_predictions(df_org, pipeline_preProcess_Standarized_pca_new, modelLGB, y_scaler, prediction_column_name='PredictedSaleprice'):

    # Transformar los datos
    new_df_org_transformed = pipeline_preProcess_Standarized_pca_new.transform(df_org)

    # Hacer predicciones
    predictions_scaled = modelLGB.predict(new_df_org_transformed)

    # Desescalar las predicciones
    predictions_descaled = y_scaler.inverse_transform(predictions_scaled.reshape(-1, 1)).ravel()

    # Verificar la coherencia de las dimensiones
    if len(predictions_descaled) != len(df_org):
        raise ValueError("El número de predicciones no coincide con el número de filas en df_org.")
    
    # Añadir las predicciones al DataFrame original
    df_org[prediction_column_name] = predictions_descaled
    
    return df_org

codigo_dir = Path(__file__).resolve().parent

anexos_dir = codigo_dir.parent.parent
data_dir = anexos_dir / "Datos"  

# Definir la ruta completa al archivo CSV
input_csv_path = data_dir / "test_distances.csv"

# Verificar la existencia del archivo
if not input_csv_path.exists():
    raise FileNotFoundError(f"No se encontró el archivo CSV en la ruta: {input_csv_path}")

df_org = pd.read_csv(input_csv_path, index_col=0)

df_org = df_org.drop('SalePrice', axis=1)

# Ruta absoluta a los archivos de modelo y pipeline
pipeline_path = Path(codigo_dir/"pkls/pipeline_preProcess_Standarized_pca.pkl")
model_path = Path(codigo_dir/"pkls/lightgbm_model.pkl")
y_scaler_path = Path(codigo_dir/"pkls/y_scaler.pkl")

# Verificar la existencia de los archivos
for path in [pipeline_path, model_path, y_scaler_path]:
    if not path.exists():
        raise FileNotFoundError(f"No se encontró el archivo en la ruta: {path}")

pipeline_preProcess_Standarized_pca_new = joblib.load(pipeline_path)
modelLGB = joblib.load(model_path)
y_scaler = joblib.load(y_scaler_path)

df_org_with_predictions = process_and_add_predictions(df_org, pipeline_preProcess_Standarized_pca_new, modelLGB, y_scaler)

# Guardar el DataFrame con las predicciones
df_org_with_predictions.to_excel(data_dir/'predictions.xlsx', index=True)
