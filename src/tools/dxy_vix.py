from fredapi import Fred

# 1. 키 설정
fred = Fred(api_key='5582bca9e6acf05ef0a4f59ed7fd7a26')

# 2. VIX (공포지수) - 최근 5개만 
print("--- [VIX] ---")
print(fred.get_series('VIXCLS').tail())

# 3. DXY Proxy (달러인덱스 대체재: 선진국 통화 대비 달러) - 최근 5개만
print("\n--- [Dollar Index (Proxy)] ---")
print(fred.get_series('DTWEXBGS').tail())


