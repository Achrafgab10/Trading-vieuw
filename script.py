import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import date

st.set_page_config(layout="wide")
st.markdown("<h1 style='text-align:center;'>BEOM Trading Tools</h1>", unsafe_allow_html=True)


tab1, tab2 = st.tabs([" Trading View", " Comparer Séquences"])

with tab1:
    mode = st.radio("Mode de chargement :", ["Depuis un chemin", "Uploader un fichier"])

    if mode == "Depuis un chemin":
        file_path = st.text_input("Chemin du fichier CSV :", "C:/Users/photo/Downloads/data_cleaned.csv")
        load_from_upload = False
    else:
        uploaded_file = st.file_uploader("Uploader un fichier CSV", type=["csv"])
        load_from_upload = True

    default_start = date(2024, 1, 12)
    default_end = date(2025, 6, 11)

    col1, col2, col3 = st.columns(3)
    with col1:
        from_date = st.date_input("De :", default_start, min_value=default_start, max_value=default_end)
    with col2:
        to_date = st.date_input("À :", default_end, min_value=default_start, max_value=default_end)
    with col3:
        interval = st.selectbox("Intervalle :", ['5min', '15min', '1h', '4h', '1d'])

    @st.cache_data
    def load_data(path_or_file, upload=False):
        df = pd.read_csv(path_or_file, sep=';', skiprows=1, names=['Time', 'Open', 'High', 'Low', 'Close'], usecols=range(5))
        df['Time'] = pd.to_datetime(df['Time'], errors='coerce', format='%d.%m.%Y %H:%M')
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

    try:
        if (not load_from_upload and file_path) or (load_from_upload and uploaded_file):
            df = load_data(file_path if not load_from_upload else uploaded_file, upload=load_from_upload)
            from_dt = pd.to_datetime(from_date)
            to_dt = pd.to_datetime(to_date) + pd.Timedelta(days=1)
            df = df[(df['Time'] >= from_dt) & (df['Time'] < to_dt)]

            if df.empty:
                st.warning("Aucune donnée disponible pour cette plage de dates.")
            else:
                df = resample_data(df, interval)

                fig = go.Figure()
                fig.add_trace(go.Candlestick(
                    x=df['Time'],
                    open=df['Open'],
                    high=df['High'],
                    low=df['Low'],
                    close=df['Close'],
                    increasing_line_color='green',
                    decreasing_line_color='red',
                    name="Prix"
                ))

                fig.update_layout(
                    height=700,
                    template='plotly_white',
                    xaxis=dict(type='date', rangeslider=dict(visible=False)),
                    yaxis=dict(fixedrange=False),
                    dragmode='pan',
                    hovermode='x unified',
                    margin=dict(t=10, b=10, l=10, r=10),
                    showlegend=False
                )

                st.plotly_chart(fig, use_container_width=True, config={
                    "scrollZoom": True,
                    "displayModeBar": False
                })

    except Exception as e:
        st.error(f"Erreur : {e}")

with tab2:
    st.markdown("<h2 style='text-align:center; color:#2c3e50;'>Comparer Deux Séquences</h2>", unsafe_allow_html=True)
    st.write("---")

    mode = st.radio("Chargement de la data :", ["Depuis un chemin", "Uploader un fichier"], key="mode_seq")

    if mode == "Depuis un chemin":
        path = st.text_input("Chemin du fichier CSV :", "C:/Users/photo/Downloads/data_cleaned.csv", key="file_path_seq")
        data_file = None
    else:
        data_file = st.file_uploader("Uploader un fichier CSV", type=["csv"], key="upload_seq")
        path = None

    @st.cache_data
    def load_data(source, upload=False):
        df = pd.read_csv(source, sep=';', skiprows=1, names=['Time', 'Open', 'High', 'Low', 'Close'], usecols=range(5))
        df['Time'] = pd.to_datetime(df['Time'], errors='coerce', format='%d.%m.%Y %H:%M')
        df.dropna(inplace=True)
        df.sort_values('Time', inplace=True)
        return df

    if ((mode == "Depuis un chemin" and path) or (mode == "Uploader un fichier" and data_file)):
        df = load_data(path if mode == "Depuis un chemin" else data_file, upload=(mode == "Uploader un fichier"))

        # Séquences valides par défaut (déjà vérifiées dans la data)
        default_start1 = pd.Timestamp("2024-01-16 09:15")
        default_end1 = pd.Timestamp("2024-01-16 09:25")
        default_start2 = pd.Timestamp("2024-01-16 10:20")
        default_end2 = pd.Timestamp("2024-01-16 10:30")

        st.subheader("Séquence 1")
        col1a, col1b = st.columns(2)
        with col1a:
            date1_start = st.date_input("Date Début 1", default_start1.date(), key="d1s")
            date1_end = st.date_input("Date Fin 1", default_end1.date(), key="d1e")
        with col1b:
            time1_start = st.time_input("Heure Début 1", default_start1.time(), key="t1s")
            time1_end = st.time_input("Heure Fin 1", default_end1.time(), key="t1e")

        st.subheader("Séquence 2")
        col2a, col2b = st.columns(2)
        with col2a:
            date2_start = st.date_input("Date Début 2", default_start2.date(), key="d2s")
            date2_end = st.date_input("Date Fin 2", default_end2.date(), key="d2e")
        with col2b:
            time2_start = st.time_input("Heure Début 2", default_start2.time(), key="t2s")
            time2_end = st.time_input("Heure Fin 2", default_end2.time(), key="t2e")

        start1 = pd.Timestamp.combine(date1_start, time1_start)
        end1 = pd.Timestamp.combine(date1_end, time1_end)
        start2 = pd.Timestamp.combine(date2_start, time2_start)
        end2 = pd.Timestamp.combine(date2_end, time2_end)

        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Comparer", key="compare_btn"):
            seq1 = df[(df['Time'] >= start1) & (df['Time'] <= end1)].copy()
            seq2 = df[(df['Time'] >= start2) & (df['Time'] <= end2)].copy()

            if seq1.empty or seq2.empty:
                st.error("❌ Une des séquences est vide. Vérifiez les plages de dates/horaires.")
            elif len(seq1) != len(seq2):
                st.error(f"❌ Les séquences doivent avoir le même nombre de points (actuel : {len(seq1)} vs {len(seq2)}).")
            else:
                similarity_ratio = (seq1['Close'].round(2).values == seq2['Close'].round(2).values).mean()
                similarity_percent = round(similarity_ratio * 100, 2)

                st.markdown(f"""
                <div style='
                    background-color: #2ecc71;
                    color: white;
                    padding: 20px;
                    text-align: center;
                    font-size: 2rem;
                    font-weight: bold;
                    border-radius: 10px;
                    margin-top: 20px;
                    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
                '>
                    ✅ Similarité : {similarity_percent} %
                </div>
                """, unsafe_allow_html=True)






