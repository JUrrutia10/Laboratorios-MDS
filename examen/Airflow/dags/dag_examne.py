from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.empty import EmptyOperator
from airflow.operators.bash import BashOperator
from airflow.utils.dates import days_ago
from ex import getdata, getdf, retraindate
from datetime import datetime

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
    schedule = None) as dag:
 
    # Task 1 - Just a simple print statement
    dummy_task = EmptyOperator(task_id='Starting_the_process', retries=2)  
    # Task 1

    # Task 2 - Download the dataset with BashOperator
    task_download_dataset = PythonOperator(
        task_id='download_dataset',
        python_callable=getdata,
        op_kwargs={
        'token': 'glpat-NJT3eVzr3oSQmk-JMEPp',
    }
      
    )


    dummy_task   >> task_download_dataset 