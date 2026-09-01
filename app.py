import streamlit as st
import numpy as np
import pandas as pd
import joblib
from keras.models import load_model

# 페이지 설정 (wide 레이아웃 및 넓어진 너비 적용)
st.set_page_config(
    page_title="PredRAIN",
    page_icon="🌧️",
    layout="wide",
    initial_sidebar_state="auto"
)

# 전체 페이지 너비를 조금 더 넓게 확장하고 마진을 준 CSS
st.markdown("""
    <style>
    .block-container {
        max-width: 1300px;
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        font-weight: 600;
        height: 3.2em;
        background-color: #2563eb;
        color: white;
    }
    .stButton>button:hover {
        background-color: #1d4ed8;
        color: white;
    }
    .guide-container {
        background-color: white;
        padding: 16px 20px;
        border-radius: 12px;
        border: 1px solid #cbd5e1;
        margin-bottom: 20px;
    }
    .result-container {
        background-color: #f1f5f9;
        padding: 24px;
        border-radius: 12px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03);
        margin-bottom: 24px;
    }
    .rain-value {
        font-size: 3rem;
        font-weight: 700;
        color: #2563eb;
        margin-top: 5px;
    }
    </style>
""", unsafe_allow_html=True)

# 타이틀 및 설명
st.markdown("""
    <h1 style='color: #1e3a8a; font-size: 2.2rem; font-weight: 700; margin-bottom: 0rem;'>AI기반 강수량 예측 서비스</h1>
""", unsafe_allow_html=True)
st.write("기상 정보를 입력하시면 AI가 강수량을 예측하고 상황에 맞게 활동 조언을 해드립니다.")
st.divider()

# 1. 모델, 스케일러, 인코더 일괄 로드
@st.cache_resource
def load_artifacts():
    model = load_model("models/precipitation_lstm_model.keras")
    scaler_feature = joblib.load("scalers/scaler_feature.pkl")
    scaler_target = joblib.load("scalers/scaler_target.pkl")
    encoder = joblib.load("encoders/encoder.pkl")
    return model, scaler_feature, scaler_target, encoder

try:
    model, scaler_feature, scaler_target, encoder = load_artifacts()
except Exception as e:
    st.error(f"필요한 파일을 불러오는 데 실패했습니다. 경로를 확인해주세요. 에러: {e}")
    st.stop()

# 샘플 데이터셋 정의
test_samples_raw = {
    "선택하세요 (직접 입력)": None,
    "맑은 날": {
        "temp": 25.0, "dew": 8.0, "humidity": 40.0, "sealevelpressure": 1020.0, 
        "cloudcover": 5.0, "precipprob": 0.0, "precipcover": 0.0, 
        "windspeed": 2.0, "windgust": 3.0, "winddir": 110.0, "preciptype": "none"
    },
    "흐린 날": {
        "temp": 18.5, "dew": 17.0, "humidity": 90.0, "sealevelpressure": 970.0, 
        "cloudcover": 100.0, "precipprob": 80.0, "precipcover": 50.0, 
        "windspeed": 4.0, "windgust": 7.0, "winddir": 180.0, "preciptype": "rain"
    },
    "비 오는 날": {
        "temp": 23.0, "dew": 20.0, "humidity": 98.0, "sealevelpressure": 940.0, 
        "cloudcover": 100.0, "precipprob": 95.0, "precipcover": 90.0, 
        "windspeed": 6.0, "windgust": 10.0, "winddir": 190.0, "preciptype": "rain"
    },
    "폭우가 쏟아지는 날": {
        "temp": 13.0, "dew": 12.9, "humidity": 100.0, "sealevelpressure": 840.0, 
        "cloudcover": 100.0, "precipprob": 70.0, "precipcover": 100.0, 
        "windspeed": 30.0, "windgust": 8.0, "winddir": 120.0, "preciptype": "rain"
    }
}

selected_preset = st.selectbox("샘플 데이터 불러오기", list(test_samples_raw.keys()))

preset_data = test_samples_raw[selected_preset]
if preset_data is None:
    preset_data = {
        "temp": 20.0, "dew": 10.0, "humidity": 60.0, "sealevelpressure": 1013.0, 
        "cloudcover": 30.0, "precipprob": 20.0, "precipcover": 0.0, 
        "windspeed": 3.0, "windgust": 5.0, "winddir": 180.0, "preciptype": "none"
    }

# 좌우 2분할 구조 생성
left_column, right_column = st.columns([1, 1], gap="large")

with left_column:
    st.subheader("기상 데이터 입력")
    st.caption("시뮬레이션을 위한 최근 기상 지표를 확인하거나 수정해 주세요.")

    temp = st.number_input("기온 (°C)", value=float(preset_data["temp"]), step=0.5)
    dew = st.number_input("이슬점 (°C)", value=float(preset_data["dew"]), step=0.5)
    humidity = st.number_input("습도 (%)", value=float(preset_data["humidity"]), min_value=0.0, max_value=100.0, step=1.0)
    sealevelpressure = st.number_input("해면기압 (hPa)", value=float(preset_data["sealevelpressure"]), step=1.0)
    cloudcover = st.number_input("구름 양 (%)", value=float(preset_data["cloudcover"]), min_value=0.0, max_value=100.0, step=1.0)
    precipprob = st.number_input("강수 확률 (%)", value=float(preset_data["precipprob"]), min_value=0.0, max_value=100.0, step=1.0)
    precipcover = st.number_input("강수 구역 비율", value=float(preset_data["precipcover"]), step=1.0)
    windspeed = st.number_input("풍속 (m/s)", value=float(preset_data["windspeed"]), step=0.5)
    windgust = st.number_input("돌풍 (m/s)", value=float(preset_data["windgust"]), step=0.5)
    winddir = st.number_input("풍향 (°)", value=float(preset_data["winddir"]), step=1.0)
    
    preciptype_options = ["none", "rain"]
    default_preciptype_idx = preciptype_options.index(preset_data["preciptype"]) if preset_data["preciptype"] in preciptype_options else 0
    preciptype = st.selectbox("강수 형태", preciptype_options, index=default_preciptype_idx)

    st.markdown("<br>", unsafe_allow_html=True)
    predict_clicked = st.button("강수량 예측 실행", type="primary")

with right_column:
    st.subheader("강수량 예측 결과")
    
    # 결과 영역 위에 마진과 간격을 둔 강수량 기준 안내 표 삽입
    st.markdown("""
        <div class="guide-container">
            <div style="font-weight: 600; font-size: 0.95rem; color: #1e293b; margin-bottom: 10px;">📊 강수량 기준 가이드</div>
            <table style="width: 100%; font-size: 0.88rem; color: #475569; border-collapse: collapse;">
                <tr style="border-bottom: 1px solid #e2e8f0;"><td style="padding: 6px 0; font-weight: 600; width: 35%;">0.0 mm</td><td style="padding: 6px 0;">맑은 날 (비 없음)</td></tr>
                <tr style="border-bottom: 1px solid #e2e8f0;"><td style="padding: 6px 0; font-weight: 600;">0.1 ~ 4.99 mm</td><td style="padding: 6px 0;">흐린 날 (약한 이슬비)</td></tr>
                <tr style="border-bottom: 1px solid #e2e8f0;"><td style="padding: 6px 0; font-weight: 600;">0.5 ~ 9.99 mm</td><td style="padding: 6px 0;">비 오는 날 (가벼운 비)</td></tr>
                <tr><td style="padding: 6px 0; font-weight: 600;">10.0 mm 이상</td><td style="padding: 6px 0;">폭우가 쏟아지는 날</td></tr>
            </table>
        </div>
    """, unsafe_allow_html=True)

    result_placeholder = st.container()

    if predict_clicked:
        numeric_cols = [
            "temp", "dew", "humidity", "sealevelpressure", "cloudcover",
            "precipprob", "precipcover", "windspeed", "windgust", "winddir"
        ]
        
        input_data = {}
        for col, val in zip(
            numeric_cols, 
            [temp, dew, humidity, sealevelpressure, cloudcover, precipprob, precipcover, windspeed, windgust, winddir]
        ):
            trend = np.linspace(-0.5, 0.0, 7) * (val * 0.05)
            input_data[col] = [max(0.0, val + t) for t in trend]
            
        input_data["preciptype"] = [preciptype] * 7
        
        input_df = pd.DataFrame(input_data)
        
        precip_encoded = encoder.transform(input_df[["preciptype"]])
        encoded_col_names = encoder.get_feature_names_out(["preciptype"])
        precip_encoded_df = pd.DataFrame(precip_encoded, columns=encoded_col_names, index=input_df.index)
        
        input_df[numeric_cols] = scaler_feature.transform(input_df[numeric_cols])
        final_input_df = pd.concat([input_df.drop(columns=["preciptype"]), precip_encoded_df], axis=1)
        X_pred = np.array([final_input_df.values], dtype=np.float32)

        pred_scaled = model.predict(X_pred, verbose=0)
        pred_log = scaler_target.inverse_transform(pred_scaled).flatten()
        final_pred = np.expm1(pred_log)[0]
        final_pred = max(0.0, final_pred)

        with result_placeholder:
            st.markdown(f"""
                <div class="result-container">
                    <div style="font-size: 1.05rem; color: #4b5563;">예상 강수량</div>
                    <div class="rain-value">{final_pred:.2f} <span style="font-size: 1.4rem; font-weight: 500; color: #6b7280;">mm</span></div>
                </div>
            """, unsafe_allow_html=True)
            
            st.markdown("#### AI 조언")
            if final_pred < 0.1:
                st.info("비가 오지 않고 맑은 날씨가 예상됩니다.")
                st.write("날씨가 쾌적하여 야외 활동을 즐기기 좋습니다. 나가서 산책을 하는 것은 어떠신가요?")
            elif final_pred < 5.0:
                st.info("구름 낀 날씨로 아주 약한 이슬비나 안개가 예상됩니다.")
                st.write("우산이 필수는 아니지만, 야외 활동 시 가볍게 챙기고 나가는 것을 권장드립니다.")
            elif final_pred < 10.0:
                st.warning("가벼운 비나 소나기가 지나갈 수 있습니다.")
                st.write("외출하실 때 작은 우산을 챙기시는 것을 추천드립니다.")
            else:
                st.error("제법 굵은 비가 내릴 것으로 예상됩니다.")
                st.write("우산을 반드시 지참하시고, 안전사고에 유의하시기 바라며 가급적 외출은 자제하시길 권장드립니다.")
    else:
        with result_placeholder:
            st.info("왼쪽에서 기상 정보를 입력하고 **강수량 예측 실행** 버튼을 눌러주세요.")