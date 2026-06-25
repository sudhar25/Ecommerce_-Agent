import sqlite3
import pandas as pd
import os

conn = sqlite3.connect('olist.db')
cursor = conn.cursor()

data_folder ='./data'

print("starting database creation...")

for file_name in os.listdir(data_folder):
    if file_name.endswith('.csv'):
        
        table_name = file_name.replace('olist_', '').replace('_dataset.csv', '').replace('.csv', '')
        
        file_path = os.path.join(data_folder, file_name)
        
        print(f"loading {file_name} into {table_name}..")
        df = pd.read_csv(file_path)
        
        
        df.to_sql(table_name, conn, if_exists='replace', index=False)
        
        
conn.commit()
conn.close()

print("\n Success all csv loded into tables")

