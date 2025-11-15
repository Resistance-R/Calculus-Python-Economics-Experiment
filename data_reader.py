import pandas as pd
import numpy as np

def import_data():
    # CSV 파일 불러오기
    data = "Bitcoin-data.csv"
    df = pd.read_csv(data)

    # 쉼표 제거 및 수치형 변환
    for col in ["종가", "시가", "고가", "저가"]:
        df[col] = df[col].astype(str).str.replace(",", "").astype(float)
    df["거래량"] = df["거래량"].apply(parse_volume)

    # 날짜 정리 (공백 제거 후 datetime 변환)
    df["날짜"] = df["날짜"].str.replace(" ", "")
    df["날짜"] = pd.to_datetime(df["날짜"], format="%Y-%m-%d")

    # 데이터 정렬 (오름차순)
    df = df.sort_values("날짜").reset_index(drop=True)

    # t를 모델 함수의 변수로 계산하기 위해 숫자형 인덱스로 변환
    df["t"] = np.arange(1, len(df)+1)

    return df # 데이터 프레입 반환

# 거래량: 'K'(천), 'M'(백만) 처리
def parse_volume(v):
    v = str(v).replace(",", "").strip()
    if v.endswith("K"):
        return float(v[:-1]) * 1_000
    elif v.endswith("M"):
        return float(v[:-1]) * 1_000_000
    else:
        return float(v)