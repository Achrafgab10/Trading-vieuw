import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import date, datetime, timedelta

st.set_page_config(layout="wide")
st.markdown("<h1 style='text-align:center;'>BEOM Trading Tools</h1>", unsafe_allow_html=True)


@st.cache_data
def load_data():
    df = pd.read_csv("data_cleaned.csv", sep=';', skiprows=1, names=['Time', 'Open', 'High', 'Low', 'Close'], usecols=range(5))
    df['Time'] = pd.to_datetime(df['Time'], format='%d.%m.%Y %H:%M', errors='coerce')
    df.dropna(inplace=True)
    df.sort_values('Time', inplace=True)
    return df.reset_index(drop=True)

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


tab1, tab2, tab3 = st.tabs(["Trading View", "Comparer Séquences", "Pattern Finder"])


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



with tab3:
    st.subheader(" Pattern Finder — Détection de Formes Similaires")

    # Inputs utilisateur avec clés uniques
    col1, col2 = st.columns(2)
    with col1:
        pattern_start_date = st.date_input(
            " Date de début du motif",
            value=datetime(2024, 1, 12).date(),
            key="pattern_start_date"
        )
        n_candles = st.number_input(
            " Nombre de chandelles",
            min_value=3, max_value=200, value=10,
            key="pattern_n_candles"
        )
    with col2:
        pattern_start_time = st.time_input(
            " Heure de début",
            value=datetime.strptime("14:20", "%H:%M").time(),
            key="pattern_start_time"
        )
        method = st.radio(
            "️ Méthode de comparaison :",
            ["Forme (z-score + corrélation)", "Valeurs réelles (écart absolu)"],
            key="comparison_method"
        )

    # Contrôle du nombre de résultats à afficher
    top_n_results = st.number_input(
        " Nombre de séquences similaires à afficher",
        min_value=1, max_value=50, value=10, step=1,
        key="top_n_results"
    )

    pattern_start_ts = datetime.combine(pattern_start_date, pattern_start_time)
    pattern_df = df[df['Time'] >= pattern_start_ts].head(n_candles).copy()

    if len(pattern_df) < n_candles:
        st.error(" Pas assez de données pour former le motif.")
        st.stop()

    # Affichage du motif sélectionné en chandeliers
    st.markdown("###  Motif sélectionné")
    fig_pattern = go.Figure(data=[go.Candlestick(
        x=pattern_df['Time'],
        open=pattern_df['Open'],
        high=pattern_df['High'],
        low=pattern_df['Low'],
        close=pattern_df['Close'],
        increasing_line_color='green',
        decreasing_line_color='red'
    )])
    fig_pattern.update_layout(height=300, margin=dict(l=10, r=10, t=10, b=10), xaxis_rangeslider_visible=False)
    st.plotly_chart(fig_pattern, use_container_width=True)

    # Fonction de normalisation z-score
    def zscore(x):
        return (x - np.mean(x)) / np.std(x) if np.std(x) != 0 else x

    pattern_raw = pattern_df['Close'].values
    pattern_norm = zscore(pattern_raw)

    # Seuil de similarité selon la méthode choisie
    if "Forme" in method:
        similarity_threshold = st.slider(
            " Seuil de corrélation (%)", 0, 100, 60, key="corr_threshold"
        )
    else:
        similarity_threshold = st.slider(
            " Écart moyen maximum (valeur réelle)", 0.0, 50.0, 5.0, key="abs_threshold"
        )

    # Recherche de séquences similaires
    results = []
    for i in range(len(df) - n_candles):
        window = df.iloc[i:i + n_candles]
        if window['Time'].iloc[0] == pattern_start_ts:
            continue

        candidate_close = window['Close'].values

        if "Forme" in method:
            candidate_norm = zscore(candidate_close)
            corr = np.corrcoef(pattern_norm, candidate_norm)[0, 1]
            if corr >= similarity_threshold / 100:
                results.append({
                    "start": window['Time'].iloc[0],
                    "end": window['Time'].iloc[-1],
                    "score": round(corr * 100, 2)
                })
        else:
            mae = np.mean(np.abs(pattern_raw - candidate_close))
            if mae <= similarity_threshold:
                results.append({
                    "start": window['Time'].iloc[0],
                    "end": window['Time'].iloc[-1],
                    "score": round(mae, 2)
                })

    # Tri des résultats
    results = sorted(results, key=lambda x: -x['score'] if "Forme" in method else x['score'])

    # Affichage des résultats
    if results:
        st.markdown(f"###  Top {min(top_n_results, len(results))} séquences similaires")
        for i, res in enumerate(results[:top_n_results]):
            score_text = f"Similarité : {res['score']}%" if "Forme" in method else f"Écart moyen : {res['score']}"
            with st.expander(f"{i+1}.  {res['start']} → {res['end']} | {score_text}", expanded=False):
                sub_df = df[(df['Time'] >= res['start']) & (df['Time'] <= res['end'])].copy()
                fig = go.Figure(data=[go.Candlestick(
                    x=sub_df['Time'],
                    open=sub_df['Open'],
                    high=sub_df['High'],
                    low=sub_df['Low'],
                    close=sub_df['Close'],
                    increasing_line_color='green',
                    decreasing_line_color='red'
                )])
                fig.update_layout(height=300, margin=dict(l=10, r=10, t=10, b=10), xaxis_rangeslider_visible=False)
                st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning(" Aucune séquence similaire trouvée.")