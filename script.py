import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import date, datetime

st.set_page_config(layout="wide")
st.markdown("<h1 style='text-align:center;'>BEOM Trading Tools</h1>", unsafe_allow_html=True)

# ------------------- Chargement direct du fichier -------------------
@st.cache_data
def load_data():
    df = pd.read_csv("data_cleaned.csv", sep=';', skiprows=1, names=['Time', 'Open', 'High', 'Low', 'Close'], usecols=range(5))
    df['Time'] = pd.to_datetime(df['Time'], format='%d.%m.%Y %H:%M', errors='coerce')
    df.dropna(inplace=True)
    df.sort_values('Time', inplace=True)
    return df

def resample_data(df, interval):
    rule = {
        '5min': '5min',
        '15min': '15min',
        '1h': '1H',
        '4h': '4H',
        '1d': '1D'
    }[interval]
    df = df.set_index('Time')
    df_resampled = df.resample(rule).agg({
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last'
    }).dropna().reset_index()
    return df_resampled

df = load_data()

# ------------------- Tabs -------------------
tab1, tab2 = st.tabs(["Trading View", "Comparer Séquences"])

# ------------------- Tab 1 -------------------
with tab1:
    col1, col2, col3 = st.columns(3)
    with col1:
        from_date = st.date_input("De :", date(2024, 1, 12))
    with col2:
        to_date = st.date_input("À :", date(2025, 6, 11))
    with col3:
        interval = st.selectbox("Intervalle :", ['5min', '15min', '1h', '4h', '1d'])

    from_dt = pd.to_datetime(from_date)
    to_dt = pd.to_datetime(to_date) + pd.Timedelta(days=1)
    df_filtered = df[(df['Time'] >= from_dt) & (df['Time'] < to_dt)]

    if df_filtered.empty:
        st.warning("Aucune donnée disponible pour cette plage de dates.")
    else:
        df_resampled = resample_data(df_filtered, interval)
        fig = go.Figure()
        fig.add_trace(go.Candlestick(
            x=df_resampled['Time'],
            open=df_resampled['Open'],
            high=df_resampled['High'],
            low=df_resampled['Low'],
            close=df_resampled['Close'],
            increasing_line_color='green',
            decreasing_line_color='red'
        ))
        fig.update_layout(
            height=700,
            xaxis_rangeslider_visible=False,
            margin=dict(l=10, r=10, t=10, b=10),
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)

# ------------------- Tab 2 -------------------
with tab2:
    st.subheader("Comparer Deux Séquences")

    col1a, col1b = st.columns(2)
    with col1a:
        date1_start = st.date_input("Date Début 1", date(2024, 1, 12))
        date1_end = st.date_input("Date Fin 1", date(2024, 1, 12))
    with col1b:
        time1_start = st.time_input("Heure Début 1", datetime.strptime("14:20", "%H:%M").time())
        time1_end = st.time_input("Heure Fin 1", datetime.strptime("14:30", "%H:%M").time())

    col2a, col2b = st.columns(2)
    with col2a:
        date2_start = st.date_input("Date Début 2", date(2025, 6, 11))
        date2_end = st.date_input("Date Fin 2", date(2025, 6, 11))
    with col2b:
        time2_start = st.time_input("Heure Début 2", datetime.strptime("12:45", "%H:%M").time())
        time2_end = st.time_input("Heure Fin 2", datetime.strptime("12:55", "%H:%M").time())

    start1 = datetime.combine(date1_start, time1_start)
    end1 = datetime.combine(date1_end, time1_end)
    start2 = datetime.combine(date2_start, time2_start)
    end2 = datetime.combine(date2_end, time2_end)

    if st.button("Comparer"):
        seq1 = df[(df['Time'] >= start1) & (df['Time'] <= end1)].copy()
        seq2 = df[(df['Time'] >= start2) & (df['Time'] <= end2)].copy()

        if len(seq1) == 0 or len(seq2) == 0:
            st.error("Une des séquences est vide.")
        elif len(seq1) != len(seq2):
            st.error(f"Les séquences doivent avoir la même longueur (actuel : {len(seq1)} vs {len(seq2)})")
        else:
            similarity = (seq1['Close'].round(2).values == seq2['Close'].round(2).values).mean() * 100
            st.success(f"Similarité entre les séquences : {similarity:.2f} %")
