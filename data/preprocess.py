import re

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def load_battery_sequence(data_csv, window_size=50):
    """
    NASA Battery Dataset의 다양한 열 조합을 자동 감지하여 시계열 윈도우를 생성하는 함수.
    상태 지표(SOC, SOH, RUL)는 별도 주어지지 않으며, decoder가 예측할 Task로 설정됨.
    모든 컬럼이 복소수인 경우, 문자열을 복소수로 변환 후 절댓값(real magnitude)만 사용함.
    """
    df = pd.read_csv(data_csv)

    # 모든 문자열 컬럼 복소수 -> 실수 절댓값 처리
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].apply(
                lambda z: abs(complex(re.sub(r'[\(\) ]', '', z))) if isinstance(z, str) else z
            )

    # 가능한 피처 조합 정의
    possible_feature_groups = [
        ['Voltage_measured', 'Current_measured', 'Temperature_measured', 'Current_load'],
        ['Sense_current', 'Battery_current', 'Current_ratio', 'Battery_impedance', 'Rectified_Impedance'],
    ]

    for group in possible_feature_groups:
        if all(col in df.columns for col in group):
            selected_features = group
            break
    else:
        # 가능한 조합 없으면 모든 수치형 컬럼을 사용
        selected_features = df.select_dtypes(include=[np.number]).columns.tolist()
        print("⚠️ 예상된 피처 조합 없음. 모든 수치형 컬럼 사용:", selected_features)

    feature_scaler = MinMaxScaler()
    X = feature_scaler.fit_transform(df[selected_features])

    X_seq = []
    for i in range(len(X) - window_size):
        X_seq.append(X[i:i+window_size])

    return np.array(X_seq, dtype=np.float32), feature_scaler

res=load_battery_sequence(data_csv='data/00002.csv', window_size=30)

print(res)
