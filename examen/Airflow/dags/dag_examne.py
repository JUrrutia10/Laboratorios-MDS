from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.empty import EmptyOperator
from airflow.operators.bash import BashOperator
from airflow.utils.dates import days_ago
from ex import getdata, getdf, retrain_model,preprocessor2,runMF, check_data_drift,predapi,rungradio
from datetime import datetime
from airflow.models import Variable

default_args = {
    'owner': 'jabier',
    'depends_on_past': False,
    'start_date': datetime(2023, 1, 1),
    'retries': 1,
}



with DAG(
    
    dag_id='my_first_pipeline', ## Name of DAG run
    default_args=default_args,
    description='MLops pipeline',
    start_date = days_ago(1),
    schedule_interval='@weekly',
    schedule = None) as dag:

    api_token = Variable.get("api_token")
    current_date = datetime.now().strftime("%Y-%m-%d")
 
    # Task 1 - Just a simple print statement
    start_task = EmptyOperator(task_id='Starting_the_process', retries=2)  
    # Task 1


    # Task 2 - Download the dataset with BashOperator
    task_download_dataset = PythonOperator(
        task_id='download_dataset',
        python_callable=getdata,
        op_kwargs={
        'token': api_token,} )
    
    
    task_get_df = PythonOperator(
    task_id='to_four_DF',
    python_callable=getdf,
    op_kwargs={
        'date': '2024-06-30',} )
    
    task_runMF = PythonOperator(
    task_id='run_MLFLOW',
    python_callable=runMF,)
    
    task_retrain = PythonOperator(
    task_id='retrain',
    python_callable=retrain_model,)



    task_gradio = PythonOperator(
    task_id='rungradio',
    python_callable=rungradio,)





    


    start_task  >> task_runMF>> task_download_dataset >>task_get_df >> task_retrain>>task_gradio




    