#%%
# Libreria Core del lab.
import numpy as np
import pandas as pd
import datetime



# Librerias utiles
from sklearn.feature_selection import VarianceThreshold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

import requests
import subprocess
import os
import joblib
import mlflow
import mlflow.xgboost
from sklearn.pipeline import Pipeline
import xgboost as xgb
import pandas as pd
import optuna
from optuna import Trial
from sklearn.metrics import precision_recall_curve, auc
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from xgboost import XGBClassifier
from datetime import datetime
import re
import pickle
from scipy.stats import ks_2samp
import gradio as gr
import shap

#%%
current_time = datetime.now().strftime("%Y-%m-%d")
# Configuración

def getdata(token):
    GITLAB_BASE_URL = 'https://gitlab.com/api/v4'
    REPO_ID = '56596242'  # Puedes encontrar este ID en la URL de tu repositorio
    REF = 'main'  # Nombre de la rama
    PATH = 'competition_files'  # Ruta de la carpeta en el repositorio
    PRIVATE_TOKEN = token  # Tu token de acceso personal

    # Obtén la lista de archivos en la carpeta
    url = f"{GITLAB_BASE_URL}/projects/{REPO_ID}/repository/tree"
    params = {
        'ref': REF,
        'path': PATH,
        'per_page': 100  # Número máximo de archivos a obtener por página
    }
    headers = {
        'Private-Token': PRIVATE_TOKEN
    }

    response = requests.get(url, headers=headers, params=params)
    response.raise_for_status()  # Verifica si la solicitud fue exitosa
    files = response.json()

    # Filtra solo los archivos CSV
    csv_files = [file['name'] for file in files if file['name'].endswith('.csv')]



    # Lee los archivos CSV desde GitLab
    data = {}
    for file in csv_files:
        file_url = f"https://gitlab.com/imezadelajara/datos_clase_7_mds7202/-/raw/main/{PATH}/{file}"


        data[file.split('.')[0]] = pd.read_csv(file_url, delimiter=',', encoding='utf-8', on_bad_lines='skip')
    for dato in data.keys():
        match = re.search(r'(\d{4}-\d{2}-\d{2})-(\w)$', dato)

        if match:
            fecha = match.group(1)
            ultimo_caracter = match.group(2)
        current_value = data[dato]
        
        # Crear un nuevo diccionario con los metadatos adicionales
        updated_value = {
            'data': current_value,
            'fecha': fecha,
            'tipo': ultimo_caracter
        }
        
        # Actualizar el diccionario original con el nuevo valor
        data[dato] = updated_value
    with open('dataex.pkl', 'wb') as f:
        pickle.dump(data, f)
#%%




#%%

def getdf(date):
    with open('dataex.pkl', 'rb') as f:
        data = pickle.load(f)

    X_old=pd.DataFrame()
    y_old=pd.DataFrame()
    X_new=pd.DataFrame()
    y_new=pd.DataFrame()

    for key, value in data.items():
    # Verificar si el valor de 'fecha' es '2024-07-12'
        if  value['fecha']<date and value['tipo'] == 'X' :
  
            X_old=pd.concat([X_old,pd.DataFrame(value['data'])], axis=0)
        elif  value['fecha']<date and value['tipo'] == 'y' :
            y_old=pd.concat([y_old,pd.DataFrame(value['data']).loc[:, 'is_mob']], axis=0)
        elif  value['fecha']>=date and value['tipo'] == 'X' :
            X_new=pd.concat([X_new,pd.DataFrame(value['data'])], axis=0)
        elif  value['fecha']>=date and value['tipo'] == 'y' :
            y_new=pd.concat([y_new,pd.DataFrame(value['data']).loc[:, 'is_mob']], axis=0)
    
    X_old.to_csv("X_old.csv")
    y_old.to_csv("y_old.csv")
    X_new.to_csv("X_new.csv")
    y_new.to_csv("y_new.csv")
        


#%%
numerical_columns=['DaysSinceJob',
 'CreditCap',
 'Speed24h',
 'AliveSession',
 'BankSpots8w',
 'HustleMinutes',
 'RiskScore',
 'AliasMatch',
 'DeviceEmails8w',
 'HustleMonth',
 'ZipHustle',
 'Speed4w',
 'income',
 'FreeMail',
 'HomePhoneCheck',
 'BankMonths',
 'DOBEmails4w',
 'ForeignHustle',
 'DeviceScams',
 'OldHoodMonths',
 'intended_balcon_amount',
 'NewCribMonths',
 'Speed6h',
 'CellPhoneCheck',
 'customer_age',
 'ExtraPlastic']

categorical_columns=['JobStatus', 'CribStatus', 'LootMethod', 'InfoSource', 'DeviceOS']
#%%

# Pipelines para columnas numéricas y categóricas con VarianceThreshold
numeric_transformations = Pipeline([
    ('scaler', MinMaxScaler()),
    ('variance_threshold', VarianceThreshold(threshold=0)) # Para eliminar DeviceScams
])

categoric_transformations = Pipeline([
    ('category_one_hot', OneHotEncoder(sparse_output=False, handle_unknown="ignore")),
    ('variance_threshold', VarianceThreshold(threshold=0))
])

# Preprocessor
preprocessor = ColumnTransformer(transformers=[
    ('numerical', numeric_transformations, numerical_columns),
    ('categorical', categoric_transformations, categorical_columns)
])
#%%
preprocessor2 = ColumnTransformer(transformers=[
    ('numerical', Pipeline([
    ('scaler', MinMaxScaler()),
    ('variance_threshold', VarianceThreshold(threshold=0)) # Para eliminar DeviceScams
]), ['DaysSinceJob',
 'CreditCap',
 'Speed24h',
 'AliveSession',
 'BankSpots8w',
 'HustleMinutes',
 'RiskScore',
 'AliasMatch',
 'DeviceEmails8w',
 'HustleMonth',
 'ZipHustle',
 'Speed4w',
 'income',
 'FreeMail',
 'HomePhoneCheck',
 'BankMonths',
 'DOBEmails4w',
 'ForeignHustle',
 'DeviceScams',
 'OldHoodMonths',
 'intended_balcon_amount',
 'NewCribMonths',
 'Speed6h',
 'CellPhoneCheck',
 'customer_age',
 'ExtraPlastic']),
    ('categorical', Pipeline([
    ('category_one_hot', OneHotEncoder(sparse_output=False, handle_unknown="ignore")),
    ('variance_threshold', VarianceThreshold(threshold=0))
]), ['JobStatus', 'CribStatus', 'LootMethod', 'InfoSource', 'DeviceOS'])
])
#%%

#%%

def optimize_hyperparameters_with_mlflow(X, y, preprocessor):
    def objective2(trial: Trial):
        # Definir los hiperparámetros a ajustar
        xgb_params = {
            'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.1, log=True),
            'n_estimators': trial.suggest_int('n_estimators', 50, 1000),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'max_leaves': trial.suggest_int('max_leaves', 0, 100),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 5),
            'reg_alpha': trial.suggest_float('reg_alpha', 0, 1),
            'reg_lambda': trial.suggest_float('reg_lambda', 0, 1),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'gamma': trial.suggest_float('gamma', 0, 5),
            'scale_pos_weight': trial.suggest_float('scale_pos_weight', 0.5, 10)
        }

        # Crear el pipeline
        XGB_pipe = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss', **xgb_params))
        ])

        # Validación cruzada
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        y_pred = cross_val_predict(XGB_pipe, X, y, cv=cv, method='predict')

        # Calcular el AUC PR
        precision, recall, _ = precision_recall_curve(y, y_pred)
        auc_pr = auc(recall, precision)

        # Rastrear los hiperparámetros y la métrica con MLFlow
        mlflow.log_params(xgb_params)
        mlflow.log_metric("auc_pr", auc_pr)

        return auc_pr

    # Ejecutar la optimización con Optuna y MLFlow
    study = optuna.create_study(direction='maximize')
    with mlflow.start_run(run_name="optimize_hyperparameters", nested=True):
        study.optimize(objective2, n_trials=10)

    # Obtener el AUC PR y los mejores hiperparámetros encontrados
    best_auc = study.best_value
    best_params = study.best_params
    num_trials = len(study.trials)
    best_model = study.best_trial

    print(f'Número de trials: {num_trials}')
    print(f'AUC PR: {best_auc}')
    print('Mejores hiperparámetros encontrados:')
    for key, value in best_params.items():
        print(f'{key}: {value}')

    return best_params,best_auc

#%%
def runMF():
    server_process = subprocess.Popen(
    ["mlflow", "server", "--host", "0.0.0.0", "--port", "5000"],
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE)
    # Set our tracking server uri for logging
    mlflow.set_tracking_uri(uri="http://mlflow:5000")

    # Create a new MLflow Experiment
    mlflow.set_experiment("MLflow Examen")

#%%



def check_data_drift(X_train, X_new, threshold=0.05):
    drift_metrics = {}
    common_columns = X_train.columns
    
    for column in common_columns:
        stat, p_value = ks_2samp(X_train[column], X_new[column])
        drift_metrics[column] = p_value
    
    # Verificar si algún valor p es menor que el umbral
    return any(p_value < threshold for p_value in drift_metrics.values())
#%%

def retrain_model(preprocessor=preprocessor2, model_path='xgb_pipeline.joblib'):
    X_old = pd.read_csv("X_old.csv", index_col=0)
    y_old = pd.read_csv("y_old.csv", index_col=0)
    X_new = pd.read_csv("X_new.csv", index_col=0)
    y_new = pd.read_csv("y_new.csv", index_col=0)
    run_name = f"incremental_training_{current_time}"

    # Configurar URI de seguimiento de MLflow
    mlflow.set_tracking_uri("http://mlflow:5000")
    mlflow.set_experiment("MLflow Examen")

    # Finalizar cualquier ejecución activa de MLFlow
    if mlflow.active_run():
        mlflow.end_run()

    # Combinar los datos antiguos y nuevos
    X_combined = pd.concat([X_old, X_new], axis=0)
    y_combined = pd.concat([y_old, y_new], axis=0)
    
    # Comprobar el drift de los datos
    datadrift = check_data_drift(X_old, X_new, threshold=0.05)
    
    def log_interpretability(X, model, run_name):
        explainer = shap.Explainer(model)
        shap_values = explainer(X)
        shap_summary_plot = shap.summary_plot(shap_values, X, show=False)
        shap.bar_plot = shap.bar_plot(shap_values, show=False)
        mlflow.log_figure(shap_summary_plot, f"shap_summary_plot_{run_name}.png")
        mlflow.log_figure(shap.bar_plot, f"shap_bar_plot_{run_name}.png")
    
    if not os.path.exists(model_path):
        with mlflow.start_run(run_name=run_name):
            try:
                # Initial training
                params, auclog = optimize_hyperparameters_with_mlflow(X_old, y_old, preprocessor)
                
                mlflow.log_params(params)
                mlflow.log_metric("aucpr", auclog)
                XGB_pipe = Pipeline([
                    ("preprocessor", preprocessor),
                    ("classifier", XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss', **params))
                ])
                XGB_pipe.fit(X_combined, y_combined)
                joblib.dump(XGB_pipe, model_path)
                mlflow.sklearn.log_model(XGB_pipe, "model_initial")
                
                # Log interpretability
                X_combined_preprocessed = XGB_pipe.named_steps['preprocessor'].transform(X_combined)
                
            finally:
                mlflow.end_run()
    else:
        if datadrift:
            with mlflow.start_run(run_name=run_name):
                try:
                    # Cargar el pipeline completo
                    XGB_pipe = joblib.load(model_path)
                    classifier = XGB_pipe.named_steps['classifier']
                    
                    # Reentrenar el modelo con los datos combinados
                    X_combined_preprocessed = XGB_pipe.named_steps['preprocessor'].transform(X_combined)
                    dtrain_combined = DMatrix(X_combined_preprocessed, label=y_combined)
                    params, auclog = optimize_hyperparameters_with_mlflow(X_combined, y_combined, preprocessor)
                    mlflow.log_params(params)
                    mlflow.log_metric("aucpr", auclog)
                    
                    classifier = xgb_train(params, dtrain_combined, num_boost_round=50, xgb_model=classifier.get_booster())
                    XGB_pipe.named_steps['classifier'] = classifier
                    joblib.dump(XGB_pipe, model_path)
                    mlflow.sklearn.log_model(XGB_pipe, "model_updated")
                    
                    
                finally:
                    mlflow.end_run()
        else:
            with mlflow.start_run(run_name=run_name):
                try:
                    # Cargar el pipeline completo
                    XGB_pipe = joblib.load(model_path)
                    X_combined_preprocessed = XGB_pipe.named_steps['preprocessor'].transform(X_combined)
                    
                    
                finally:
                    mlflow.end_run()
                
    return model_path
#%%

def log_shap_interpretability(model_path='xgb_pipeline.joblib', sample_size=100):
    # Cargar los datos combinados y el modelo
    X_combined = pd.concat([pd.read_csv("X_old.csv", index_col=0), pd.read_csv("X_new.csv", index_col=0)], axis=0)
    XGB_pipe = joblib.load(model_path)
    run_name = f"shap_interpretability_{datetime.now().strftime('%Y%m%d')}"

    # Configurar URI de seguimiento de MLflow
    mlflow.set_tracking_uri("http://mlflow:5000")
    mlflow.set_experiment("MLflow Examen")

    # Tomar una muestra de los datos
    X_sample = X_combined.sample(sample_size)
    X_sample_preprocessed = XGB_pipe.named_steps['preprocessor'].transform(X_sample)
    
    with mlflow.start_run(run_name=run_name):
        # Calcular los valores SHAP
        explainer = shap.Explainer(XGB_pipe.named_steps['classifier'])
        shap_values = explainer(X_sample_preprocessed)
        
        # Crear y guardar los gráficos de SHAP
        shap_summary_plot = shap.summary_plot(shap_values, X_sample_preprocessed, show=False)
        shap_bar_plot = shap.bar_plot(shap_values, show=False)
        
        mlflow.log_figure(shap_summary_plot, "shap_summary_plot.png")
        mlflow.log_figure(shap_bar_plot, "shap_bar_plot.png")

        mlflow.end_run()

#%%

def retraindate(date):
    first_run='2024-06-26'
    try:
        if last_run<date:
            runMF()
            getdf(date)
            last_model=retrain_model()
            last_run=date
        return last_run,last_model
    except:
        'no hay lastrun'

    if first_run<date:
        runMF()
        getdf(date)
        last_model=retrain_model()
        last_run=date
    return last_run,last_model


#%%

#last_model=retrain_model(datalist1, preprocessor)


#%%
def predapi(data,model_path = 'xgb_pipeline.joblib'):
    # Iniciar una nueva ejecución de MLflow
    with mlflow.start_run():
        # Registrar el momento de la ejecución
        mlflow.log_param("execution_time", datetime.now().isoformat())

        # Leer los datos de entrada
        datadf = pd.read_csv(data)
        
        # Registrar el número de filas en los datos de entrada
        mlflow.log_param("num_rows", len(datadf))
        model = joblib.load(model_path)
        # Realizar la predicción
        ypred = model.predict(datadf)
        ypred_df = pd.DataFrame(ypred, columns=["Prediction"])
        dataypred = datadf.copy()
        dataypred['label_predicted'] = ypred

        # Registrar los resultados de la predicción como un artefacto
        prediction_path = "/tmp/predictions.csv"
        ypred_df.to_csv(prediction_path, index=False)
        mlflow.log_artifact(prediction_path)

        # Opcional: registrar algunas métricas de las predicciones
        # Por ejemplo, la media de las predicciones
        mlflow.log_metric("mean_prediction", ypred_df["Prediction"].mean())

    return ypred_df
#%%
def rungradio():
    demo = gr.Interface(fn = predapi, 
                        inputs=gr.File(type="filepath"), 
                        outputs=gr.DataFrame()) # valor de salida

    demo.launch(share = True)
#%%

# %%
