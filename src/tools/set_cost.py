import requests
from bs4 import BeautifulSoup
import numpy as np

# ----- Forward ------
# Spot 환율

url = "https://api.freecurrencyapi.com/v1/latest?apikey=fca_live_EzkBRnnng2e9jMy7suF8fHiQrwt991Gu7a7LSu2t"
response = requests.get(url)
data = response.json()

spot_price_today = data["data"]["KRW"]
print("현재 스팟 가격:", spot_price_today)

# Spot Prices
url = "https://www.daishin.com/g.ds?m=1071&p=2536&v=1875"
headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
response = requests.get(url, headers=headers)
soup = BeautifulSoup(response.text, "html.parser")

selected_divs = []

for td in soup.find_all("td"):
    td_class = td.get("class", [])  # type: ignore
    if td_class == ["first"] or td_class == ["right"]:
        div = td.find("div", class_="tdArea")  # type: ignore
        if div:
            selected_divs.append(div.get_text(strip=True))

exchange_rate = [float(r.replace(',', '')) if i % 2 == 1 else r for i, r in enumerate(selected_divs)]

spot_prices = np.array(exchange_rate[1::2])  # 환율만 추출
dates = np.array(exchange_rate[0::2])       # 날짜만 추출

# Interest rate (한국, 미국)

apikey = "NLRL7KPYC2KZ33PX3Z8P"
STAT_CODE = "902Y006"

# 2025년 4월 ~ 2025년 11월
months = [f"{y}{m:02d}" for y in range(2025, 2026) for m in range(4, 12)]

r_krw_list = []
r_usd_list = []

prev_r_krw = None
prev_r_usd = None

for month in months:
    url_rate = f'https://ecos.bok.or.kr/api/StatisticSearch/{apikey}/json/kr/1/100/{STAT_CODE}/M/{month}/{month}'
    response = requests.get(url_rate)
    data = response.json()

    r_krw = None
    r_usd = None

    if 'StatisticSearch' in data and 'row' in data['StatisticSearch']:
        for item in data['StatisticSearch']['row']:
            if item['ITEM_NAME1'] == '한국':
                r_krw = float(item['DATA_VALUE']) / 100
            elif item['ITEM_NAME1'] == '미국':
                r_usd = float(item['DATA_VALUE']) / 100

    # 데이터 없으면 이전 값으로 대체
    if r_krw is None:
        r_krw = prev_r_krw
    if r_usd is None:
        r_usd = prev_r_usd

    r_krw_list.append(r_krw)
    r_usd_list.append(r_usd)

    prev_r_krw = r_krw
    prev_r_usd = r_usd

# Forward Price
T = 1/12  # 1개월 만기

spot_months = [d[:7] for d in dates]  # 'YYYY/MM' 형식

month_to_r_krw = {month: r for month, r in zip(months, r_krw_list)}
month_to_r_usd = {month: r for month, r in zip(months, r_usd_list)}

r_krw_daily = np.array([month_to_r_krw[m.replace('/', '')] for m in spot_months])
r_usd_daily = np.array([month_to_r_usd[m.replace('/', '')] for m in spot_months])

r_krw_today = r_krw_daily[0]
r_usd_today = r_usd_daily[0]

forward_prices = spot_prices * np.exp((r_krw_daily - r_usd_daily) * T)
forward_price_today = spot_price_today * np.exp((r_krw_today - r_usd_today) * T)

# hedge ratio 계산 
delta_S = np.log(spot_prices[1:] / spot_prices[:-1])
delta_F = np.log(forward_prices[1:] / forward_prices[:-1])
# delta_S = np.diff(spot_prices)
# delta_F = np.diff(forward_prices)

h_star = np.cov(delta_S, delta_F)[0, 1] / np.var(delta_F)
h_star = min(h_star, 1)
print("Minimum Variance Hedge Ratio (h*):", h_star)

# USD 1,000,000 환선도 헤지 비용
exposure = 1_000_000
notional = h_star * exposure
hedge_cost = notional * forward_price_today
print("FX Forward 비용 (KRW):", hedge_cost)

# ----- Options -----

from math import log, sqrt, exp, erf

def norm_cdf(x):
    return 0.5 * (1 + erf(x / sqrt(2)))

def fx_put_price_gk_vectorized(S, K, r_d, r_f, sigma, T):
    d1 = (log(S / K) + (r_d - r_f + 0.5 * sigma ** 2) * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)
    put_price = K * exp(-r_d * T) * norm_cdf(-d2) - S * exp(-r_f * T) * norm_cdf(-d1)
    delta_put = -exp(-r_f * T) * norm_cdf(-d1)
    return put_price, delta_put

# Historical Volatility

spot_prices_vol = spot_prices[:30]
returns = np.log(spot_prices_vol[1:] / spot_prices_vol[:-1])

sigma_daily = np.std(returns)
sigma_annual = sigma_daily * np.sqrt(252)
print(f"Historical Volatility: {sigma_annual:.4f}")


# -------------------------
# Example
# -------------------------
S = spot_price_today          # Spot (KRW/USD)
K = 1480          # Strike (직접 입력 필요)
sigma = sigma_annual      # Implied Volatility
r_d = r_krw       # KRW interest rate (domestic)
r_f = r_usd       # USD interest rate (foreign)
T = 1/12           # Maturity (1 year)

price, delta = fx_put_price_gk_vectorized(S, K, r_d, r_f, sigma, T)
print("풋 옵션 비용 (KRW per USD):", price)
print("헤지 비율:", abs(delta))

# USD 1,000,000 환옵션 헤지 비용
exposure = 1_000_000
notional = exposure * abs(delta)
hedge_cost = price * notional
print("FX Option 비용 (KRW):", hedge_cost)