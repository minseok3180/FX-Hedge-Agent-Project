import os
import pandas as pd
import pymysql
import numpy as np
from dotenv import load_dotenv

# .env 로드
load_dotenv()

DB_HOST = os.getenv("DB_HOST")
DB_PORT = int(os.getenv("DB_PORT", 3306))
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")

# CSV 읽기
df = pd.read_csv("/Users/minseok/FX-Hedge-Agent-Project/notebook/yonju/df.csv")

# DB 연결
conn = pymysql.connect(
    host=DB_HOST,
    port=DB_PORT,
    user=DB_USER,
    password=DB_PASSWORD,
    database=DB_NAME,
    charset="utf8mb4"
)
cursor = conn.cursor()

sql = """
INSERT INTO eiExchangeRate (
    date, usdkrw, us_ex, us_im, reserve, us_reserve, us_export, us_import, 
    base, market, consumer, exp_rate, im_rate, us_current, us_growth, us_gdp, us_stock, us_interest
) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
"""

for _, row in df.iterrows():
    values = tuple(row.replace({np.nan: None}))
    cursor.execute(sql, values)

conn.commit()
cursor.close()
conn.close()

print("✅ CSV 데이터 업로드 완료!")
