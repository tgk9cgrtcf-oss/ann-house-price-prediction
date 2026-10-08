
import streamlit as st

st.set_page_config(
    page_title="AI 주택 가격 예측 연구소",
    page_icon="🏠",
    layout="wide"
)

st.title("AI 주택 가격 예측 연구소")
st.subheader("인공신경망 기반 캘리포니아 주택 가격 예측")

st.write(
    "공개 데이터를 활용하여 인공신경망을 학습시키고, "
    "주택 가격을 예측하는 데이터 분석 프로그램입니다."
)

st.divider()

st.header("프로젝트 소개")
st.write("데이터: California Housing Dataset")
st.write("인공지능 알고리즘: MLPRegressor")
st.write("분석 방법: 인공신경망과 선형회귀 모델 비교")
st.write("평가 지표: R², RMSE, MAE")
import pandas as pd
from sklearn.datasets import fetch_california_housing

st.divider()
st.header("01. 공개 데이터 탐색")

@st.cache_data
def load_data():
    housing = fetch_california_housing(as_frame=True)
    return housing.frame

df = load_data()

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("전체 데이터 수", f"{len(df):,}개")

with col2:
    st.metric("입력 변수", "8개")

with col3:
    st.metric("예측 목표", "주택 가격")

st.subheader("실제 데이터 미리보기")
st.markdown(df.head(20).to_html(index=False), unsafe_allow_html=True)

st.subheader("데이터 기초 통계")
st.markdown(df.describe().round(2).to_html(), unsafe_allow_html=True)

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import numpy as np

st.divider()
st.header("02. 인공신경망 학습 및 예측")

X = df.drop(columns=["MedHouseVal"])
y = df["MedHouseVal"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

@st.cache_resource
def train_models(X_train_scaled, y_train, X_train):
    ann = MLPRegressor(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        solver="adam",
        max_iter=500,
        early_stopping=True,
        random_state=42
    )
    ann.fit(X_train_scaled, y_train)

    linear = LinearRegression()
    linear.fit(X_train, y_train)

    return ann, linear

with st.spinner("인공신경망을 학습하고 있습니다..."):
    ann_model, linear_model = train_models(
        X_train_scaled, y_train, X_train
    )

ann_pred = ann_model.predict(X_test_scaled)
linear_pred = linear_model.predict(X_test)

st.success("인공신경망 학습 완료")

st.write("인공신경망 구조: 입력층 8개 → 은닉층 64개 → 은닉층 32개 → 출력층 1개")
st.write(f"학습 데이터: {len(X_train):,}개")
st.write(f"테스트 데이터: {len(X_test):,}개")

st.divider()
st.header("03. AI 예측 성능 비교")

def calculate_metrics(actual, predicted):
    return {
        "R²": r2_score(actual, predicted),
        "RMSE": np.sqrt(mean_squared_error(actual, predicted)),
        "MAE": mean_absolute_error(actual, predicted)
    }

ann_metrics = calculate_metrics(y_test, ann_pred)
linear_metrics = calculate_metrics(y_test, linear_pred)

col1, col2 = st.columns(2)

with col1:
    st.subheader("인공신경망 ANN")
    st.metric("R² 결정계수", f"{ann_metrics['R²']:.4f}")
    st.metric("RMSE", f"{ann_metrics['RMSE']:.4f}")
    st.metric("MAE", f"{ann_metrics['MAE']:.4f}")

with col2:
    st.subheader("선형회귀 Linear Regression")
    st.metric("R² 결정계수", f"{linear_metrics['R²']:.4f}")
    st.metric("RMSE", f"{linear_metrics['RMSE']:.4f}")
    st.metric("MAE", f"{linear_metrics['MAE']:.4f}")

st.info(
    "R²는 1에 가까울수록 좋고, "
    "RMSE와 MAE는 0에 가까울수록 예측 오차가 작습니다."
)

import matplotlib.pyplot as plt

st.divider()
st.header("04. 실제값과 AI 예측값 비교")

fig, ax = plt.subplots(figsize=(9, 6))

ax.scatter(
    y_test,
    ann_pred,
    alpha=0.35,
    s=15,
    label="ANN Predictions"
)

min_value = min(y_test.min(), ann_pred.min())
max_value = max(y_test.max(), ann_pred.max())

ax.plot(
    [min_value, max_value],
    [min_value, max_value],
    linestyle="--",
    color="red",
    label="Perfect Prediction"
)

ax.set_xlabel("Actual House Value (100,000 USD)")
ax.set_ylabel("Predicted House Value (100,000 USD)")
ax.set_title("ANN: Actual vs Predicted")
ax.legend()
ax.grid(alpha=0.2)

st.pyplot(fig)
plt.close(fig)

st.write(
    "점이 빨간 점선에 가까울수록 실제 주택 가격과 "
    "인공신경망의 예측 가격이 비슷하다는 의미입니다."
)

st.divider()
st.header("05. AI 주택 가격 직접 예측")

st.write(
    "주택이 위치한 지역의 정보를 입력하면 "
    "학습된 인공신경망이 주택 가격을 예측합니다."
)

input_values = {}

with st.form("prediction_form"):
    col1, col2 = st.columns(2)

    for i, feature in enumerate(X.columns):
        with col1 if i % 2 == 0 else col2:
            input_values[feature] = st.number_input(
                feature,
                value=float(X[feature].median()),
                format="%.4f"
            )

    predict_button = st.form_submit_button(
        "AI로 주택 가격 예측하기",
        use_container_width=True
    )

if predict_button:
    input_df = pd.DataFrame([input_values])
    input_scaled = scaler.transform(input_df)

    predicted_value = ann_model.predict(input_scaled)[0]

    st.success("AI 예측이 완료되었습니다!")

    st.metric(
        "예상 주택 가격",
        f"${predicted_value * 100000:,.0f}"
    )

    st.caption(
        "예측값은 California Housing 데이터의 지역별 "
        "중위 주택 가격을 기준으로 합니다. "
        "개별 주택의 실제 거래 가격과는 다를 수 있습니다."
    )

st.divider()
st.header("06. 입력 변수 설명")

feature_descriptions = {
    "MedInc": "지역의 중위 가구 소득 (단위: 1만 달러)",
    "HouseAge": "지역 내 주택의 중위 연식 (년)",
    "AveRooms": "가구당 평균 방 개수",
    "AveBedrms": "가구당 평균 침실 개수",
    "Population": "해당 지역의 인구",
    "AveOccup": "가구당 평균 거주 인원",
    "Latitude": "지역의 위도",
    "Longitude": "지역의 경도"
}

description_df = pd.DataFrame(
    [
        {"변수명": name, "설명": description}
        for name, description in feature_descriptions.items()
    ]
)

st.markdown(
    description_df.to_html(index=False),
    unsafe_allow_html=True
)

st.caption(
    "California Housing 데이터는 개별 주택이 아닌 "
    "캘리포니아의 인구조사 구역별 정보를 담고 있습니다."
)

st.divider()
st.header("07. 인공신경망 알고리즘 구조")

st.write(
    "본 프로그램은 다층 퍼셉트론(MLP) 인공신경망을 "
    "사용하여 주택 가격을 예측합니다."
)

layers = [
    ("입력층", "8개 변수"),
    ("은닉층 1", "64개 뉴런"),
    ("은닉층 2", "32개 뉴런"),
    ("출력층", "주택 가격 1개")
]

cols = st.columns(4)

for col, (name, detail) in zip(cols, layers):
    with col:
        st.info(f"{name}\n\n{detail}")

st.subheader("인공신경망 학습 과정")

st.write("1. 공개 데이터에서 8개의 입력 변수를 추출합니다.")
st.write("2. 데이터를 학습용 80%, 테스트용 20%로 분리합니다.")
st.write("3. 입력 변수의 크기를 표준화합니다.")
st.write("4. 은닉층에서 가중치와 편향을 이용해 값을 계산합니다.")
st.write("5. ReLU 활성화 함수를 적용합니다.")
st.write("6. 예측 오차를 바탕으로 역전파를 수행합니다.")
st.write("7. Adam 최적화 알고리즘으로 가중치를 갱신합니다.")
st.write("8. 테스트 데이터로 예측 성능을 평가합니다.")

st.divider()
st.header("08. 인공신경망 예측 오차 분석")

residuals = y_test.to_numpy() - ann_pred

fig_error, ax_error = plt.subplots(figsize=(9, 5))

ax_error.scatter(
    ann_pred,
    residuals,
    alpha=0.35,
    s=15
)

ax_error.axhline(
    y=0,
    color="red",
    linestyle="--",
    label="Zero Error"
)

ax_error.set_xlabel("Predicted House Value (100,000 USD)")
ax_error.set_ylabel("Residual: Actual - Predicted")
ax_error.set_title("ANN Prediction Residuals")
ax_error.legend()
ax_error.grid(alpha=0.2)

st.pyplot(fig_error)
plt.close(fig_error)

st.write(
    "빨간 점선은 예측 오차가 0인 기준선입니다. "
    "점들이 기준선 주변에 고르게 분포할수록 "
    "특정 가격대에서 발생하는 예측 편향이 적다는 의미입니다."
)
st.divider()
st.header("09. 인공신경망 학습 손실 분석")

loss_values = ann_model.loss_curve_

fig_loss, ax_loss = plt.subplots(figsize=(10, 5))
ax_loss.plot(range(1, len(loss_values) + 1), loss_values)
ax_loss.set_xlabel("Training Iteration")
ax_loss.set_ylabel("Loss")
ax_loss.set_title("ANN Training Loss Curve")
ax_loss.grid(True, alpha=0.3)

st.pyplot(fig_loss)
plt.close(fig_loss)

st.write(
    "이 그래프는 인공신경망이 학습하는 동안 손실 값이 "
    "어떻게 변화했는지 보여줍니다. "
    "손실 값이 감소하면 학습 데이터에 대한 오차가 "
    "줄어들고 있다는 의미입니다."
)
st.divider()
st.header("10. 인공신경망 구조별 성능 비교 실험")

st.write(
    "은닉층의 뉴런 수와 구조를 변경하면서 "
    "인공신경망의 예측 성능이 어떻게 달라지는지 비교합니다."
)

if st.button("인공신경망 구조 비교 실험 실행"):
    structures = {
        "은닉층 1개 (16)": (16,),
        "은닉층 2개 (32, 16)": (32, 16),
        "은닉층 2개 (64, 32)": (64, 32)
    }

    experiment_results = []

    with st.spinner("인공신경망 구조별 실험을 진행하고 있습니다..."):
        for name, structure in structures.items():
            experiment_model = MLPRegressor(
                hidden_layer_sizes=structure,
                activation="relu",
                solver="adam",
                max_iter=500,
                early_stopping=True,
                random_state=42
            )

            experiment_model.fit(X_train_scaled, y_train)
            experiment_pred = experiment_model.predict(X_test_scaled)

            experiment_results.append({
                "인공신경망 구조": name,
                "R²": r2_score(y_test, experiment_pred),
                "RMSE": mean_squared_error(
                    y_test, experiment_pred
                ) ** 0.5,
                "MAE": mean_absolute_error(y_test, experiment_pred)
            })

    results_df = pd.DataFrame(experiment_results)

    st.text(results_df.round(4).to_string(index=False))

    fig_compare, ax_compare = plt.subplots(figsize=(10, 5))
    ax_compare.bar(
        ["1 Layer (16)", "2 Layers (32, 16)", "2 Layers (64, 32)"],
        results_df["R²"]
    )
    ax_compare.set_xlabel("ANN Structure")
    ax_compare.set_ylabel("R² Score")
    ax_compare.set_title("ANN Structure Performance Comparison")
    ax_compare.set_ylim(0, 1)
    plt.xticks(rotation=15)
    plt.tight_layout()

    st.pyplot(fig_compare)
    plt.close(fig_compare)

    best_model = results_df.loc[results_df["R²"].idxmax()]

    st.success(
        f"이번 실험에서 가장 높은 R²를 기록한 구조는 "
        f"{best_model['인공신경망 구조']}입니다. "
        f"R² = {best_model['R²']:.4f}"
    )

    st.write(
        "은닉층의 구조가 복잡하다고 반드시 성능이 좋아지는 것은 아닙니다. "
        "모델의 구조와 데이터의 특성에 따라 예측 성능이 달라질 수 있습니다."
    )