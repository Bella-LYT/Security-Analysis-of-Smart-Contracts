from google.cloud import bigquery
from google.oauth2 import service_account
bytecode = [] #字节码数据
swc_id = []  #标签数据
def retrive_dataset():  #python连接bigquery表（自己在bigquery上建的）并读取数据
    credentials = service_account.Credentials.from_service_account_file('path') #凭据路径(json文件)
    project_id = 'citric-cistern-420507' #Google project名称
    client = bigquery.Client(credentials= credentials,project=project_id)

    query_job = client.query("""
        SELECT bytecode,swc_id FROM `blockchain.label` 
        """)
    results = query_job.result() 
    for row in results:
        bytecode.append(row[0])
        swc_id.append(row[1])