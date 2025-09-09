import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import date, datetime, timedelta
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import random
import warnings
from typing import Optional, Tuple, List, Dict, Any

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Optimize page config for performance
st.set_page_config(
    page_title="BEOM Trading Tools",
    page_icon="📈",
    layout="wide", 
    initial_sidebar_state="collapsed",
    menu_items={
        'About': "Advanced Trading Analysis Tools with AI-powered Pattern Recognition"
    }
)

# Enhanced styling
st.markdown("""
<style>
.main .block-container {
    padding-top: 1rem;
    padding-bottom: 1rem;
}
.stMetric {
    background-color: rgba(28, 131, 225, 0.1);
    border: 1px solid rgba(28, 131, 225, 0.1);
    padding: 0.5rem;
    border-radius: 0.5rem;
}
</style>
""", unsafe_allow_html=True)

st.markdown("<h1 style='text-align:center; color:#1c83e1;'>📈 BEOM Trading Tools</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:#666;'>Advanced AI-Powered Trading Analysis Platform</p>", unsafe_allow_html=True)

# Performance optimization: Cache data loading with TTL
@st.cache_data(ttl=300)  # Cache for 5 minutes
def load_data():
    """Load and preprocess data with optimizations"""
    try:
        # Load CSV with proper format
        df = pd.read_csv(
            "data_cleaned.csv", 
            sep=';',
            dtype={'Open': 'float32', 'High': 'float32', 'Low': 'float32', 'Close': 'float32'}
        )
        
        # Parse datetime with correct format
        df['Time'] = pd.to_datetime(df['Time'], format='%d.%m.%Y %H:%M', errors='coerce')
        
        # Remove any rows with NaN values
        df = df.dropna()
        
        # Sort by time
        df = df.sort_values('Time').reset_index(drop=True)
        
        # Log data info for debugging (only show in sidebar or as info)
        if len(df) > 0:
            st.sidebar.success(f"✅ Data loaded: {len(df)} points ({df['Time'].min().strftime('%Y-%m-%d')} to {df['Time'].max().strftime('%Y-%m-%d')})")
        else:
            st.sidebar.error("❌ No data loaded")
        
        return df
        
    except Exception as e:
        st.error(f"❌ Error loading data: {str(e)}")
        # Return empty dataframe as fallback
        return pd.DataFrame(columns=['Time', 'Open', 'High', 'Low', 'Close'])

# Cache resampling operations
@st.cache_data(ttl=300)
def resample_data(df_filtered, interval):
    """Cached resampling function - now takes filtered data directly"""
    rule = {
        '5min': '5min',
        '15min': '15min',
        '1h': '1H',
        '4h': '4H',
        '1d': '1D'
    }[interval]
    df_indexed = df_filtered.set_index('Time')
    df_resampled = df_indexed.resample(rule).agg({
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last'
    }).dropna().reset_index()
    return df_resampled

# Optimized zscore function using numpy
@st.cache_data
def zscore(x):
    """Fast z-score calculation"""
    x = np.asarray(x, dtype=np.float32)
    std_val = np.std(x)
    return (x - np.mean(x)) / std_val if std_val != 0 else x

# Cache filtered data operations
@st.cache_data
def filter_data_by_date(df_hash, from_dt, to_dt):
    """Cached date filtering"""
    df = st.session_state.df_data
    return df[(df['Time'] >= from_dt) & (df['Time'] < to_dt)]


# Initialize data in session state for better performance
if 'df_data' not in st.session_state:
    with st.spinner('Loading data...'):
        st.session_state.df_data = load_data()
        
df = st.session_state.df_data


tab1, tab2, tab3, tab4 = st.tabs(["Trading View", "Compare Sequences", "Pattern Finder", "AI Prediction"])


with tab1:
    st.subheader("📈 Trading View")

    # Load and display data
    data_path = "data_cleaned.csv"
    df = pd.read_csv(data_path, sep=';')
    df['Time'] = pd.to_datetime(df['Time'], format='%d.%m.%Y %H:%M')

    # Date range and interval selection
    col1, col2 = st.columns(2)

    with col1:
        start_date = st.date_input("From", df['Time'].min().date())
        end_date = st.date_input("To", df['Time'].max().date())

    with col2:
        interval = st.selectbox("Interval", ['5min', '15min', '1h', '4h', '1d'])

    # Filter data based on date range
    mask = (df['Time'].dt.date >= start_date) & (df['Time'].dt.date <= end_date)
    df_filtered = df.loc[mask]

    # Resample data based on interval
    rule_map = {
        '5min': '5min',
        '15min': '15min',
        '1h': '1H',
        '4h': '4H',
        '1d': '1D'
    }
    df_resampled = df_filtered.set_index('Time').resample(rule_map[interval]).agg({
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last'
    }).dropna().reset_index()

    # Create base candlestick chart
    fig = go.Figure()
    
    # Add candlestick chart
    fig.add_trace(go.Candlestick(
        x=df_resampled['Time'],
        open=df_resampled['Open'],
        high=df_resampled['High'],
        low=df_resampled['Low'],
        close=df_resampled['Close'],
        increasing_line_color='green',
        decreasing_line_color='red',
        increasing_fillcolor='rgba(0,255,0,0.3)',
        decreasing_fillcolor='rgba(255,0,0,0.3)',
        line=dict(width=1)
    ))
    


    # Update layout
    fig.update_layout(
        title="Trading View Chart",
        xaxis_title="Time",
        yaxis_title="Price",
        xaxis_rangeslider_visible=False
    )

    # Display chart
    st.plotly_chart(fig, use_container_width=True)
    
    # Add basic statistics
    if len(df_resampled) > 0:
        st.markdown("### 📊 Period Statistics")
        
        # Calculate metrics
        first_price = df_resampled['Close'].iloc[0]
        last_price = df_resampled['Close'].iloc[-1]
        price_change = last_price - first_price
        price_change_pct = (price_change / first_price) * 100
        
        high_price = df_resampled['High'].max()
        low_price = df_resampled['Low'].min()
        price_range = high_price - low_price
        
        # Display metrics in columns
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Price Change", 
                f"{price_change:+.2f}",
                f"{price_change_pct:+.2f}%"
            )
            
        with col2:
            st.metric(
                "Period High", 
                f"{high_price:.2f}",
                f"+{((high_price - first_price) / first_price * 100):+.2f}%"
            )
            
        with col3:
            st.metric(
                "Period Low", 
                f"{low_price:.2f}",
                f"{((low_price - first_price) / first_price * 100):+.2f}%"
            )
            
        with col4:
            st.metric(
                "Price Range", 
                f"{price_range:.2f}",
                f"{(price_range / first_price * 100):.2f}% of start price"
            )


with tab2:
    st.subheader("Compare Two Trading Sequences")
    
    # Enhanced UI layout
    st.markdown("### 📊 Select Sequences to Compare")
    
    # Quick test buttons
    st.markdown("**🚀 Quick Test Presets:**")
    col_preset1, col_preset2, col_preset3 = st.columns(3)
    
    with col_preset1:
        if st.button("📈 Positive Correlation Test", key="pos_test"):
            st.session_state.seq1_start_date = date(2024, 1, 15)
            st.session_state.seq1_start_time = datetime.strptime("01:20", "%H:%M").time()
            st.session_state.seq1_end_date = date(2024, 1, 15)
            st.session_state.seq1_end_time = datetime.strptime("02:00", "%H:%M").time()
            st.session_state.seq2_start_date = date(2024, 1, 15)
            st.session_state.seq2_start_time = datetime.strptime("02:15", "%H:%M").time()
            st.session_state.seq2_end_date = date(2024, 1, 15)
            st.session_state.seq2_end_time = datetime.strptime("02:55", "%H:%M").time()
            st.rerun()
    
    with col_preset2:
        if st.button("📉 Negative Correlation Test", key="neg_test"):
            st.session_state.seq1_start_date = date(2024, 1, 12)
            st.session_state.seq1_start_time = datetime.strptime("14:20", "%H:%M").time()
            st.session_state.seq1_end_date = date(2024, 1, 12)
            st.session_state.seq1_end_time = datetime.strptime("15:00", "%H:%M").time()
            st.session_state.seq2_start_date = date(2024, 1, 15)
            st.session_state.seq2_start_time = datetime.strptime("02:15", "%H:%M").time()
            st.session_state.seq2_end_date = date(2024, 1, 15)
            st.session_state.seq2_end_time = datetime.strptime("02:55", "%H:%M").time()
            st.rerun()
    
    with col_preset3:
        if st.button("🔄 Same Period Test", key="same_test"):
            st.session_state.seq1_start_date = date(2024, 1, 15)
            st.session_state.seq1_start_time = datetime.strptime("08:00", "%H:%M").time()
            st.session_state.seq1_end_date = date(2024, 1, 15)
            st.session_state.seq1_end_time = datetime.strptime("08:40", "%H:%M").time()
            st.session_state.seq2_start_date = date(2024, 1, 15)
            st.session_state.seq2_start_time = datetime.strptime("08:00", "%H:%M").time()
            st.session_state.seq2_end_date = date(2024, 1, 15)
            st.session_state.seq2_end_time = datetime.strptime("08:40", "%H:%M").time()
            st.rerun()
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**📉 Sequence 1 (Declining)**")
        date1_start = st.date_input("Start Date 1", date(2024, 1, 15), key="seq1_start_date")
        time1_start = st.time_input("Start Time 1", datetime.strptime("10:15", "%H:%M").time(), key="seq1_start_time")
        date1_end = st.date_input("End Date 1", date(2024, 1, 15), key="seq1_end_date")
        time1_end = st.time_input("End Time 1", datetime.strptime("10:55", "%H:%M").time(), key="seq1_end_time")
        
    with col2:
        st.markdown("**📉 Sequence 2 (Declining)**")
        date2_start = st.date_input("Start Date 2", date(2024, 1, 16), key="seq2_start_date")
        time2_start = st.time_input("Start Time 2", datetime.strptime("00:20", "%H:%M").time(), key="seq2_start_time")
        date2_end = st.date_input("End Date 2", date(2024, 1, 16), key="seq2_end_date")
        time2_end = st.time_input("End Time 2", datetime.strptime("01:00", "%H:%M").time(), key="seq2_end_time")
    
    # Comparison method selection
    st.markdown("### ⚙️ Comparison Settings")
    comparison_method = st.selectbox(
        "Choose comparison method:",
        ["Price Correlation", "Exact Price Match", "Movement Direction", "Normalized Shape"]
    )
    
    # Tolerance setting for exact match
    if comparison_method == "Exact Price Match":
        tolerance = st.slider("Price tolerance", 0.01, 1.0, 0.1, 0.01)
    
    start1 = datetime.combine(date1_start, time1_start)
    end1 = datetime.combine(date1_end, time1_end)
    start2 = datetime.combine(date2_start, time2_start)
    end2 = datetime.combine(date2_end, time2_end)
    
    # Validation checks
    validation_errors = []
    if start1 >= end1:
        validation_errors.append("❌ Sequence 1: Start time must be before end time")
    if start2 >= end2:
        validation_errors.append("❌ Sequence 2: Start time must be before end time")
    
    if validation_errors:
        for error in validation_errors:
            st.error(error)
    else:
        if st.button("🔍 Compare Sequences", type="primary"):
            # Extract sequences
            seq1 = df[(df['Time'] >= start1) & (df['Time'] <= end1)].copy()
            seq2 = df[(df['Time'] >= start2) & (df['Time'] <= end2)].copy()
            
            # Data validation
            if len(seq1) == 0:
                st.error("❌ Sequence 1 is empty. No data found for the selected time range.")
            elif len(seq2) == 0:
                st.error("❌ Sequence 2 is empty. No data found for the selected time range.")
            else:
                # Display sequence info
                col_info1, col_info2 = st.columns(2)
                with col_info1:
                    st.info(f"📊 Sequence 1: {len(seq1)} data points")
                with col_info2:
                    st.info(f"📊 Sequence 2: {len(seq2)} data points")
                
                # Handle different sequence lengths
                if len(seq1) != len(seq2):
                    st.warning(f"⚠️ Sequences have different lengths: {len(seq1)} vs {len(seq2)}")
                    
                    # Option to trim to same length
                    trim_option = st.radio(
                        "How to handle different lengths?",
                        ["Trim to shorter length", "Skip comparison"],
                        horizontal=True
                    )
                    
                    if trim_option == "Skip comparison":
                        st.stop()
                    else:
                        min_len = min(len(seq1), len(seq2))
                        seq1 = seq1.head(min_len)
                        seq2 = seq2.head(min_len)
                        st.info(f"✂️ Trimmed both sequences to {min_len} points")
                
                # Perform comparison based on selected method
                if comparison_method == "Price Correlation":
                    if len(seq1) < 2:
                        st.error("❌ Need at least 2 data points for correlation analysis")
                    else:
                        correlation = np.corrcoef(seq1['Close'].values, seq2['Close'].values)[0, 1]
                        if np.isnan(correlation):
                            correlation = 0.0
                        
                        correlation_pct = correlation * 100
                        
                        # Color code the result
                        if abs(correlation_pct) >= 80:
                            st.success(f"🎯 **High Correlation**: {correlation_pct:.2f}%")
                        elif abs(correlation_pct) >= 50:
                            st.warning(f"📊 **Medium Correlation**: {correlation_pct:.2f}%")
                        else:
                            st.info(f"📈 **Low Correlation**: {correlation_pct:.2f}%")
                        
                        # Additional statistics
                        st.markdown("**📈 Statistical Summary:**")
                        col_stat1, col_stat2 = st.columns(2)
                        with col_stat1:
                            st.metric("Seq 1 Mean Price", f"{seq1['Close'].mean():.2f}")
                            st.metric("Seq 1 Price Range", f"{seq1['Close'].max() - seq1['Close'].min():.2f}")
                        with col_stat2:
                            st.metric("Seq 2 Mean Price", f"{seq2['Close'].mean():.2f}")
                            st.metric("Seq 2 Price Range", f"{seq2['Close'].max() - seq2['Close'].min():.2f}")
                
                elif comparison_method == "Exact Price Match":
                    matches = np.abs(seq1['Close'].values - seq2['Close'].values) <= tolerance
                    similarity = matches.mean() * 100
                    
                    if similarity >= 80:
                        st.success(f"🎯 **High Similarity**: {similarity:.2f}% (tolerance: ±{tolerance})")
                    elif similarity >= 50:
                        st.warning(f"📊 **Medium Similarity**: {similarity:.2f}% (tolerance: ±{tolerance})")
                    else:
                        st.info(f"📈 **Low Similarity**: {similarity:.2f}% (tolerance: ±{tolerance})")
                    
                    avg_diff = np.mean(np.abs(seq1['Close'].values - seq2['Close'].values))
                    st.metric("Average Price Difference", f"{avg_diff:.3f}")
                
                elif comparison_method == "Movement Direction":
                    if len(seq1) < 2:
                        st.error("❌ Need at least 2 data points for movement analysis")
                    else:
                        moves1 = np.sign(np.diff(seq1['Close'].values))
                        moves2 = np.sign(np.diff(seq2['Close'].values))
                        direction_match = (moves1 == moves2).mean() * 100
                        
                        if direction_match >= 80:
                            st.success(f"🎯 **High Direction Match**: {direction_match:.2f}%")
                        elif direction_match >= 50:
                            st.warning(f"📊 **Medium Direction Match**: {direction_match:.2f}%")
                        else:
                            st.info(f"📈 **Low Direction Match**: {direction_match:.2f}%")
                        
                        # Movement statistics
                        up_moves1 = (moves1 > 0).sum()
                        down_moves1 = (moves1 < 0).sum()
                        up_moves2 = (moves2 > 0).sum()
                        down_moves2 = (moves2 < 0).sum()
                        
                        col_move1, col_move2 = st.columns(2)
                        with col_move1:
                            st.metric("Seq 1 Up Moves", up_moves1)
                            st.metric("Seq 1 Down Moves", down_moves1)
                        with col_move2:
                            st.metric("Seq 2 Up Moves", up_moves2)
                            st.metric("Seq 2 Down Moves", down_moves2)
                
                elif comparison_method == "Normalized Shape":
                    # Normalize both sequences using z-score
                    norm1 = zscore(seq1['Close'].values)
                    norm2 = zscore(seq2['Close'].values)
                    
                    # Calculate correlation of normalized shapes
                    shape_corr = np.corrcoef(norm1, norm2)[0, 1]
                    if np.isnan(shape_corr):
                        shape_corr = 0.0
                    
                    shape_corr_pct = shape_corr * 100
                    
                    if abs(shape_corr_pct) >= 80:
                        st.success(f"🎯 **High Shape Similarity**: {shape_corr_pct:.2f}%")
                    elif abs(shape_corr_pct) >= 50:
                        st.warning(f"📊 **Medium Shape Similarity**: {shape_corr_pct:.2f}%")
                    else:
                        st.info(f"📈 **Low Shape Similarity**: {shape_corr_pct:.2f}%")
                
                # Visual comparison
                st.markdown("### 📊 Visual Comparison")
                col_chart1, col_chart2 = st.columns(2)
                
                with col_chart1:
                    st.markdown("**Sequence 1**")
                    fig1 = go.Figure()
                    fig1.add_trace(go.Candlestick(
                        x=seq1['Time'],
                        open=seq1['Open'],
                        high=seq1['High'],
                        low=seq1['Low'],
                        close=seq1['Close'],
                        increasing_line_color='green',
                        decreasing_line_color='red'
                    ))
                    fig1.update_layout(
                        height=400,
                        xaxis_rangeslider_visible=False,
                        margin=dict(l=10, r=10, t=30, b=10),
                        showlegend=False,
                        title=f"Sequence 1 ({len(seq1)} points)"
                    )
                    st.plotly_chart(fig1, use_container_width=True)
                
                with col_chart2:
                    st.markdown("**Sequence 2**")
                    fig2 = go.Figure()
                    fig2.add_trace(go.Candlestick(
                        x=seq2['Time'],
                        open=seq2['Open'],
                        high=seq2['High'],
                        low=seq2['Low'],
                        close=seq2['Close'],
                        increasing_line_color='green',
                        decreasing_line_color='red'
                    ))
                    fig2.update_layout(
                        height=400,
                        xaxis_rangeslider_visible=False,
                        margin=dict(l=10, r=10, t=30, b=10),
                        showlegend=False,
                        title=f"Sequence 2 ({len(seq2)} points)"
                    )
                    st.plotly_chart(fig2, use_container_width=True)



with tab3:
    st.subheader("🔍 Advanced Pattern Finder — Find Top Similar Sequences")
    
    # Create a compact UI layout
    col1, col2, col3 = st.columns([2, 2, 1])
    
    with col1:
        st.markdown("**📅 Reference Pattern**")
        pattern_start_date = st.date_input(
            "Start Date",
            value=datetime(2024, 1, 15).date(),
            key="pattern_start_date"
        )
        pattern_start_time = st.time_input(
            "Start Time",
            value=datetime.strptime("10:00", "%H:%M").time(),
            key="pattern_start_time"
        )
        
    with col2:
        st.markdown("**⚙️ Search Parameters**")
        n_candles = st.number_input(
            "Pattern Length (candles)",
            min_value=5, max_value=50, value=15,
            key="pattern_n_candles"
        )
        search_method = st.selectbox(
            "Comparison Method",
            ["Shape Correlation (Normalized)", "Price Correlation (Raw)", "Multi-Feature (OHLC)"],
            key="search_method"
        )
        
    with col3:
        st.markdown("**📊 Results**")
        top_n_results = st.number_input(
            "Top Results",
            min_value=5, max_value=20, value=10,
            key="top_n_results"
        )
        min_correlation = st.slider(
            "Min Similarity %",
            10, 95, 60,
            key="min_correlation"
        )
    
    # Create reference pattern
    pattern_start_ts = datetime.combine(pattern_start_date, pattern_start_time)
    pattern_df = df[df['Time'] >= pattern_start_ts].head(n_candles).copy()
    
    if len(pattern_df) < n_candles:
        st.error(f"❌ Not enough data. Found {len(pattern_df)} candles, need {n_candles}.")
        st.stop()
    
    # Display reference pattern
    st.markdown("### 📈 Reference Pattern")
    fig_pattern = go.Figure()
    fig_pattern.add_trace(go.Candlestick(
        x=pattern_df['Time'],
        open=pattern_df['Open'],
        high=pattern_df['High'],
        low=pattern_df['Low'],
        close=pattern_df['Close'],
        increasing_line_color='#00ff00',
        decreasing_line_color='#ff0000',
        name="Reference Pattern"
    ))
    fig_pattern.update_layout(
        height=350, 
        margin=dict(l=10, r=10, t=30, b=10), 
        xaxis_rangeslider_visible=False,
        title=f"Reference Pattern: {pattern_df['Time'].iloc[0].strftime('%Y-%m-%d %H:%M')} ({n_candles} candles)"
    )
    st.plotly_chart(fig_pattern, use_container_width=True)
    
    # Start search button
    if st.button("🚀 Find Similar Patterns", type="primary", use_container_width=True):
        
        # Advanced pattern matching algorithm
        @st.cache_data(ttl=60)
        def find_top_similar_patterns(pattern_data, search_method, min_corr, top_n):
            """Advanced pattern matching with multiple similarity metrics"""
            
            results = []
            total_windows = len(df) - n_candles + 1
            step_size = max(1, total_windows // 2000)  # Limit to 2000 comparisons max
            
            # Prepare reference pattern features
            if search_method == "Shape Correlation (Normalized)":
                ref_close = zscore(pattern_data['Close'].values.astype(np.float32))
                ref_feature = ref_close
            elif search_method == "Price Correlation (Raw)":
                ref_feature = pattern_data['Close'].values.astype(np.float32)
            else:  # Multi-Feature
                ref_close = zscore(pattern_data['Close'].values.astype(np.float32))
                ref_high = zscore(pattern_data['High'].values.astype(np.float32))
                ref_low = zscore(pattern_data['Low'].values.astype(np.float32))
                ref_volume = zscore(pattern_data['High'].values - pattern_data['Low'].values)  # Range as proxy
            
            # Progress tracking
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            processed = 0
            for i in range(0, total_windows, step_size):
                if processed % 100 == 0:
                    progress_bar.progress(min(processed / (total_windows // step_size), 1.0))
                    status_text.text(f"Analyzing pattern {processed}/{total_windows // step_size}...")
                
                window = df.iloc[i:i + n_candles]
                
                # Skip if not enough data or same as reference
                if (len(window) < n_candles or 
                    window['Time'].iloc[0] == pattern_start_ts):
                    processed += 1
                    continue
                
                try:
                    # Calculate similarity based on method
                    if search_method == "Shape Correlation (Normalized)":
                        candidate_feature = zscore(window['Close'].values.astype(np.float32))
                        correlation = np.corrcoef(ref_feature, candidate_feature)[0, 1]
                        
                    elif search_method == "Price Correlation (Raw)":
                        candidate_feature = window['Close'].values.astype(np.float32)
                        correlation = np.corrcoef(ref_feature, candidate_feature)[0, 1]
                        
                    else:  # Multi-Feature
                        cand_close = zscore(window['Close'].values.astype(np.float32))
                        cand_high = zscore(window['High'].values.astype(np.float32))
                        cand_low = zscore(window['Low'].values.astype(np.float32))
                        cand_volume = zscore(window['High'].values - window['Low'].values)
                        
                        # Combined correlation score
                        corr_close = np.corrcoef(ref_close, cand_close)[0, 1]
                        corr_high = np.corrcoef(ref_high, cand_high)[0, 1]
                        corr_low = np.corrcoef(ref_low, cand_low)[0, 1]
                        corr_volume = np.corrcoef(ref_volume, cand_volume)[0, 1]
                        
                        # Weighted average (Close price gets more weight)
                        correlation = (0.5 * corr_close + 0.2 * corr_high + 
                                     0.2 * corr_low + 0.1 * corr_volume)
                    
                    # Skip NaN correlations
                    if np.isnan(correlation):
                        processed += 1
                        continue
                    
                    correlation_pct = correlation * 100
                    
                    # Only keep high-quality matches
                    if correlation_pct >= min_corr:
                        # Calculate additional metrics
                        price_change_ref = ((pattern_data['Close'].iloc[-1] - pattern_data['Close'].iloc[0]) / 
                                          pattern_data['Close'].iloc[0] * 100)
                        price_change_cand = ((window['Close'].iloc[-1] - window['Close'].iloc[0]) / 
                                           window['Close'].iloc[0] * 100)
                        
                        volatility = np.std(window['Close'].values) / np.mean(window['Close'].values) * 100
                        
                        results.append({
                            "start": window['Time'].iloc[0],
                            "end": window['Time'].iloc[-1],
                            "similarity": round(correlation_pct, 2),
                            "price_change": round(price_change_cand, 2),
                            "ref_price_change": round(price_change_ref, 2),
                            "volatility": round(volatility, 2),
                            "avg_price": round(window['Close'].mean(), 2),
                            "data": window.copy()
                        })
                    
                except (ValueError, IndexError, ZeroDivisionError):
                    pass
                
                processed += 1
            
            # Clean up progress indicators
            progress_bar.empty()
            status_text.empty()
            
            # Sort by similarity and return top results
            return sorted(results, key=lambda x: x['similarity'], reverse=True)[:top_n]
        
        # Execute the search
        with st.spinner(f"🔍 Searching for top {top_n_results} similar patterns..."):
            similar_patterns = find_top_similar_patterns(
                pattern_df, search_method, min_correlation, top_n_results
            )
        
        # Display results
        if similar_patterns:
            st.success(f"✅ Found {len(similar_patterns)} similar patterns!")
            
            # Summary statistics
            avg_similarity = np.mean([p['similarity'] for p in similar_patterns])
            max_similarity = max([p['similarity'] for p in similar_patterns])
            
            col_stat1, col_stat2, col_stat3 = st.columns(3)
            with col_stat1:
                st.metric("Average Similarity", f"{avg_similarity:.1f}%")
            with col_stat2:
                st.metric("Best Match", f"{max_similarity:.1f}%")
            with col_stat3:
                st.metric("Total Matches", len(similar_patterns))
            
            st.markdown("### 🎯 Top Similar Patterns")
            
            # Display each result
            for i, pattern in enumerate(similar_patterns):
                similarity_color = "🟢" if pattern['similarity'] >= 80 else "🟡" if pattern['similarity'] >= 70 else "🔵"
                
                with st.expander(
                    f"{similarity_color} #{i+1} — {pattern['start'].strftime('%Y-%m-%d %H:%M')} | "
                    f"Similarity: {pattern['similarity']}% | Price Change: {pattern['price_change']:+.1f}%",
                    expanded=i < 3  # Auto-expand top 3
                ):
                    # Pattern metrics
                    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
                    with col_m1:
                        st.metric("Similarity", f"{pattern['similarity']}%")
                    with col_m2:
                        st.metric("Price Change", f"{pattern['price_change']:+.1f}%")
                    with col_m3:
                        st.metric("Volatility", f"{pattern['volatility']:.1f}%")
                    with col_m4:
                        st.metric("Avg Price", f"{pattern['avg_price']:.2f}")
                    
                    # Pattern chart
                    fig_match = go.Figure()
                    fig_match.add_trace(go.Candlestick(
                        x=pattern['data']['Time'],
                        open=pattern['data']['Open'],
                        high=pattern['data']['High'],
                        low=pattern['data']['Low'],
                        close=pattern['data']['Close'],
                        increasing_line_color='#00cc44',
                        decreasing_line_color='#ff4444',
                        name=f"Match {i+1}"
                    ))
                    fig_match.update_layout(
                        height=300, 
                        margin=dict(l=10, r=10, t=30, b=10),
                        xaxis_rangeslider_visible=False,
                        title=f"Pattern Match {i+1} — {pattern['similarity']}% similar"
                    )
                    st.plotly_chart(fig_match, use_container_width=True)
            
        else:
            st.warning(f"❌ No patterns found with similarity >= {min_correlation}%. Try lowering the minimum similarity threshold.")
            st.info("💡 **Tips**: Lower the minimum similarity or try a different comparison method.")


with tab4:
    st.subheader("AI Price Prediction - Advanced Pattern-Based Forecasting")
    
    # Configuration section
    st.markdown("### Prediction Configuration")
    
    col1, col2, col3 = st.columns([2, 2, 1])
    
    with col1:
        st.markdown("**Select Prediction Date**")
        # Check if we have data
        if len(df) == 0:
            st.error("No data available. Please check the data file.")
            st.stop()
        
        # Calculate a good default date (70% through the dataset)
        data_start = df['Time'].min().date()
        data_end = df['Time'].max().date()
        total_days = (data_end - data_start).days
        default_offset = int(total_days * 0.7)  # 70% through the data
        default_date = data_start + timedelta(days=default_offset)
        
        prediction_date = st.date_input(
            "Reference Date", 
            value=default_date, 
            min_value=data_start + timedelta(days=1),
            max_value=data_end - timedelta(days=1),
            help="The system will analyze 20 candles before this date"
        )
        prediction_time = st.time_input(
            "Reference Time", 
            value=datetime.strptime("10:00", "%H:%M").time(),
            help="Exact time for the prediction reference point"
        )
        
    with col2:
        st.markdown("**Analysis Parameters**")
        similarity_method = st.selectbox(
            "Similarity Algorithm",
            [
                "Multi-Feature Ensemble (Best)",
                "Shape Correlation (Normalized)", 
                "Price Correlation (Raw)",
                "Advanced DTW (Dynamic Time Warping)"
            ],
            help="Algorithm used to find similar patterns"
        )
        min_similarity_threshold = st.slider(
            "Minimum Similarity %", 
            40, 95, 65,
            help="Minimum similarity required for pattern matching"
        )
        
    with col3:
        st.markdown("**Results**")
        num_patterns = st.number_input(
            "Number of Patterns", 
            min_value=3, max_value=15, value=5,
            help="How many similar patterns to use for prediction"
        )
        show_confidence = st.checkbox("Show Confidence Metrics", True)
        show_individual_predictions = st.checkbox("Show Individual Matches", True)
        show_backtesting = st.checkbox("Show Backtesting Results", True, help="Test the algorithm on historical data to measure real accuracy")
    
    # Prediction button
    prediction_datetime = datetime.combine(prediction_date, prediction_time)
    
    # Add validation before the button
    reference_data_preview = df[df['Time'] < prediction_datetime].tail(20).copy()
    available_data_count = len(reference_data_preview)
    
    if available_data_count < 20:
        st.error(f"Insufficient historical data. Found only {available_data_count} candles, need 20. Please select a later date.")
        st.info(f"**Suggestion**: Try selecting a date after {df['Time'].iloc[19].strftime('%Y-%m-%d')} to ensure enough historical data.")
    else:
        st.success(f"Ready: {available_data_count} historical candles available for analysis.")
    
    if st.button("Generate AI Prediction", type="primary", use_container_width=True, disabled=(available_data_count < 20)):
        
        # Get the 20 candles before prediction date
        reference_data = df[df['Time'] < prediction_datetime].tail(20).copy()
        
        if len(reference_data) < 20:
            st.error(f"Insufficient historical data. Found {len(reference_data)} candles, need 20. Please select a later date.")
            st.stop()
        
        # Get actual future data (next 20 candles after prediction date) for validation if available
        future_data = df[(df['Time'] >= prediction_datetime)].head(20).copy()
        
        # Display reference pattern
        st.markdown("### Reference Pattern (Last 20 Candles)")
        
        # Calculate reference pattern metrics
        ref_price_change = ((reference_data['Close'].iloc[-1] - reference_data['Close'].iloc[0]) / reference_data['Close'].iloc[0]) * 100
        ref_volatility = np.std(reference_data['Close'].values) / np.mean(reference_data['Close'].values) * 100
        ref_avg_range = (reference_data['High'] - reference_data['Low']).mean()
        
        col_metrics1, col_metrics2, col_metrics3 = st.columns(3)
        with col_metrics1:
            st.metric("Price Change", f"{ref_price_change:+.2f}%")
        with col_metrics2:
            st.metric("Volatility", f"{ref_volatility:.2f}%")
        with col_metrics3:
            st.metric("Avg Range", f"{ref_avg_range:.2f}")
        
        # Display reference pattern chart
        fig_ref = go.Figure()
        fig_ref.add_trace(go.Candlestick(
            x=reference_data['Time'],
            open=reference_data['Open'],
            high=reference_data['High'],
            low=reference_data['Low'],
            close=reference_data['Close'],
            increasing=dict(
                line=dict(color='#00AA00', width=3),
                fillcolor='rgba(0, 170, 0, 0.7)'
            ),
            decreasing=dict(
                line=dict(color='#FF0000', width=3),
                fillcolor='rgba(255, 0, 0, 0.7)'
            ),
            name="Reference Pattern"
        ))
        fig_ref.update_layout(
            height=400,
            title=f"Reference Pattern: {reference_data['Time'].iloc[0].strftime('%Y-%m-%d %H:%M')} to {reference_data['Time'].iloc[-1].strftime('%Y-%m-%d %H:%M')}",
            xaxis_rangeslider_visible=False,
            margin=dict(l=20, r=20, t=50, b=20),
            xaxis=dict(
                type='date',
                tickmode='auto',
                showgrid=True,
                gridcolor='rgba(128, 128, 128, 0.2)'
            ),
            yaxis=dict(
                autorange=True,
                showgrid=True,
                gridcolor='rgba(128, 128, 128, 0.2)'
            )
        )
        st.plotly_chart(fig_ref, use_container_width=True)
        
        # Advanced pattern matching and prediction
        @st.cache_data(ttl=300)
        def find_similar_patterns_and_predict(ref_data, similarity_method, min_similarity, num_patterns):
            """Find top N similar patterns and generate predictions"""
            
            matches = []
            total_windows = len(df) - 40  # Need 20 for pattern + 20 for future
            step_size = max(1, total_windows // 3000)  # Optimize search
            
            # Prepare reference features based on method
            ref_close = ref_data['Close'].values.astype(np.float64)
            ref_high = ref_data['High'].values.astype(np.float64)
            ref_low = ref_data['Low'].values.astype(np.float64)
            ref_open = ref_data['Open'].values.astype(np.float64)
            
            def calculate_similarity(ref_data, candidate_data, method):
                """Calculate similarity using specified method"""
                cand_close = candidate_data['Close'].values.astype(np.float64)
                cand_high = candidate_data['High'].values.astype(np.float64)
                cand_low = candidate_data['Low'].values.astype(np.float64)
                cand_open = candidate_data['Open'].values.astype(np.float64)
                
                try:
                    if method == "Shape Correlation (Normalized)":
                        ref_norm = zscore(ref_close)
                        cand_norm = zscore(cand_close)
                        return np.corrcoef(ref_norm, cand_norm)[0, 1] * 100
                    
                    elif method == "Price Correlation (Raw)":
                        return np.corrcoef(ref_close, cand_close)[0, 1] * 100
                    
                    elif method == "Advanced DTW (Dynamic Time Warping)":
                        # Simplified DTW implementation
                        ref_norm = zscore(ref_close)
                        cand_norm = zscore(cand_close)
                        n, m = len(ref_norm), len(cand_norm)
                        
                        # Create DTW matrix
                        dtw_matrix = np.full((n + 1, m + 1), np.inf)
                        dtw_matrix[0, 0] = 0
                        
                        for i in range(1, n + 1):
                            for j in range(1, m + 1):
                                cost = abs(ref_norm[i-1] - cand_norm[j-1]) ** 2
                                dtw_matrix[i, j] = cost + min(
                                    dtw_matrix[i-1, j],
                                    dtw_matrix[i, j-1],
                                    dtw_matrix[i-1, j-1]
                                )
                        
                        distance = dtw_matrix[n, m]
                        max_distance = np.var(ref_norm) + np.var(cand_norm) + 1e-8
                        similarity = max(0, 100 - (distance / (max_distance * max(n, m)) * 20))
                        return similarity
                    
                    else:  # Multi-Feature Ensemble (Best)
                        # Multiple correlation features
                        correlations = []
                        
                        # 1. Normalized price correlation
                        ref_norm = zscore(ref_close)
                        cand_norm = zscore(cand_close)
                        correlations.append(abs(np.corrcoef(ref_norm, cand_norm)[0, 1]))
                        
                        # 2. High/Low correlation
                        correlations.append(abs(np.corrcoef(zscore(ref_high), zscore(cand_high))[0, 1]))
                        correlations.append(abs(np.corrcoef(zscore(ref_low), zscore(cand_low))[0, 1]))
                        
                        # 3. Range correlation (volatility pattern)
                        ref_range = ref_high - ref_low
                        cand_range = cand_high - cand_low
                        correlations.append(abs(np.corrcoef(zscore(ref_range), zscore(cand_range))[0, 1]))
                        
                        # 4. Returns correlation
                        ref_returns = np.diff(ref_close) / ref_close[:-1]
                        cand_returns = np.diff(cand_close) / cand_close[:-1]
                        correlations.append(abs(np.corrcoef(ref_returns, cand_returns)[0, 1]))
                        
                        # 5. Directional consistency
                        ref_direction = np.sign(np.diff(ref_close))
                        cand_direction = np.sign(np.diff(cand_close))
                        direction_match = (ref_direction == cand_direction).mean()
                        correlations.append(direction_match)
                        
                        # 6. Volatility pattern similarity (NEW)
                        ref_volatility = np.std(ref_close)
                        cand_volatility = np.std(cand_close)
                        vol_similarity = 1 - abs(ref_volatility - cand_volatility) / (ref_volatility + cand_volatility + 1e-8)
                        correlations.append(vol_similarity)
                        
                        # Filter out NaN values
                        valid_correlations = [c for c in correlations if not np.isnan(c)]
                        
                        if len(valid_correlations) == 0:
                            return 0
                        
                        # Enhanced weighted ensemble score with more features
                        base_weights = [0.25, 0.12, 0.12, 0.18, 0.13, 0.05, 0.15]
                        weights = np.array(base_weights[:len(valid_correlations)])
                        weights = weights / np.sum(weights)  # Normalize
                        
                        ensemble_score = np.average(valid_correlations, weights=weights) * 100
                        return ensemble_score
                        
                except:
                    return 0
            
            # Progress tracking
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            processed = 0
            for i in range(20, total_windows, step_size):
                if processed % 200 == 0:
                    progress = min(processed / (total_windows // step_size), 1.0)
                    progress_bar.progress(progress)
                    status_text.text(f"Analyzing patterns... {processed}/{total_windows // step_size}")
                
                # Get candidate pattern (20 candles)
                candidate_pattern = df.iloc[i:i + 20]
                # Get future data after pattern (20 candles)
                candidate_future = df.iloc[i + 20:i + 40]
                
                if len(candidate_pattern) < 20 or len(candidate_future) < 20:
                    processed += 1
                    continue
                
                # Skip if overlapping with our reference period
                if (candidate_pattern['Time'].iloc[-1] >= ref_data['Time'].iloc[0] and 
                    candidate_pattern['Time'].iloc[0] <= ref_data['Time'].iloc[-1]):
                    processed += 1
                    continue
                
                # Calculate similarity
                similarity_score = calculate_similarity(ref_data, candidate_pattern, similarity_method)
                
                # Additional quality filter: check if patterns have reasonable data
                pattern_quality = True
                if (candidate_pattern['Close'].std() == 0 or 
                    candidate_future['Close'].std() == 0 or
                    candidate_pattern['Close'].isnull().any() or
                    candidate_future['Close'].isnull().any()):
                    pattern_quality = False
                
                if similarity_score >= min_similarity and pattern_quality:
                    # Calculate pattern metrics
                    pattern_change = ((candidate_pattern['Close'].iloc[-1] - candidate_pattern['Close'].iloc[0]) / candidate_pattern['Close'].iloc[0]) * 100
                    future_change = ((candidate_future['Close'].iloc[-1] - candidate_pattern['Close'].iloc[-1]) / candidate_pattern['Close'].iloc[-1]) * 100
                    
                    pattern_volatility = np.std(candidate_pattern['Close'].values) / np.mean(candidate_pattern['Close'].values) * 100
                    
                    matches.append({
                        'start_time': candidate_pattern['Time'].iloc[0],
                        'end_time': candidate_pattern['Time'].iloc[-1],
                        'similarity': round(similarity_score, 2),
                        'pattern_change': round(pattern_change, 2),
                        'future_change': round(future_change, 2),
                        'volatility': round(pattern_volatility, 2),
                        'pattern_data': candidate_pattern.copy(),
                        'future_data': candidate_future.copy()
                    })
                
                processed += 1
            
            progress_bar.empty()
            status_text.empty()
            
            # Return top N matches sorted by similarity
            return sorted(matches, key=lambda x: x['similarity'], reverse=True)[:num_patterns]
        
        # Execute the search
        with st.spinner(f"🔍 Searching for similar patterns using {similarity_method}..."):
            top_matches = find_similar_patterns_and_predict(
                reference_data, 
                similarity_method, 
                min_similarity_threshold,
                num_patterns
            )
        
        if not top_matches:
            st.warning(f"❌ No similar patterns found with similarity >= {min_similarity_threshold}%. Try lowering the similarity threshold.")
            st.info("💡 **Tip**: Lower the minimum similarity or try a different algorithm.")
            st.stop()
        
        # Display results
        st.success(f"✅ Found {len(top_matches)} high-quality pattern matches!")
        
        # Generate ensemble prediction
        st.markdown("### AI Ensemble Prediction")
        
        # Calculate weighted average prediction (IMPROVED METHOD)
        total_weight = 0
        weighted_future_changes = []
        weighted_returns = []  # Store relative returns instead of absolute prices
        
        # Get the last known price as our starting point
        last_known_price = reference_data['Close'].iloc[-1]
        
        for match in top_matches:
            weight = match['similarity'] / 100.0  # Convert percentage to weight
            total_weight += weight
            weighted_future_changes.append(match['future_change'] * weight)
            
            # Calculate relative returns from the pattern's future data
            pattern_last_price = match['pattern_data']['Close'].iloc[-1]
            future_data_match = match['future_data']
            
            # Calculate returns for each candle relative to the pattern's end
            returns_sequence = []
            prev_price = pattern_last_price
            
            for _, row in future_data_match.iterrows():
                # Calculate OHLC as percentage changes from previous close
                open_return = (row['Open'] - prev_price) / prev_price
                high_return = (row['High'] - prev_price) / prev_price  
                low_return = (row['Low'] - prev_price) / prev_price
                close_return = (row['Close'] - prev_price) / prev_price
                
                returns_sequence.append({
                    'open_return': open_return,
                    'high_return': high_return, 
                    'low_return': low_return,
                    'close_return': close_return,
                    'weight': weight
                })
                prev_price = row['Close']  # Update for next iteration
            
            weighted_returns.append(returns_sequence)
        
        # Generate prediction by applying weighted average returns to last known price
        prediction_candles = []
        current_price = last_known_price
        
        for candle_idx in range(20):
            # Aggregate weighted returns for this candle position
            total_open_return = 0
            total_high_return = 0 
            total_low_return = 0
            total_close_return = 0
            total_candle_weight = 0
            
            for pattern_returns in weighted_returns:
                if candle_idx < len(pattern_returns):
                    candle_data = pattern_returns[candle_idx]
                    weight = candle_data['weight']
                    
                    total_open_return += candle_data['open_return'] * weight
                    total_high_return += candle_data['high_return'] * weight
                    total_low_return += candle_data['low_return'] * weight 
                    total_close_return += candle_data['close_return'] * weight
                    total_candle_weight += weight
            
            if total_candle_weight > 0:
                # Apply weighted average returns to current price
                pred_open = current_price * (1 + total_open_return / total_candle_weight)
                pred_high = current_price * (1 + total_high_return / total_candle_weight)
                pred_low = current_price * (1 + total_low_return / total_candle_weight)
                pred_close = current_price * (1 + total_close_return / total_candle_weight)
                
                # Ensure OHLC logic (High >= max(O,C), Low <= min(O,C))
                pred_high = max(pred_high, pred_open, pred_close)
                pred_low = min(pred_low, pred_open, pred_close)
                
                prediction_candles.append([pred_open, pred_high, pred_low, pred_close])
                current_price = pred_close  # Update for next candle
            else:
                # Fallback: use last known values
                prediction_candles.append([current_price, current_price, current_price, current_price])
        
        # Calculate ensemble prediction change
        if total_weight > 0:
            ensemble_future_change = sum(weighted_future_changes) / total_weight
        else:
            ensemble_future_change = 0
        
        # Display prediction summary with enhanced metrics
        avg_similarity = np.mean([m['similarity'] for m in top_matches])
        min_similarity = min([m['similarity'] for m in top_matches])
        bullish_count = sum(1 for m in top_matches if m['future_change'] > 0)
        bearish_count = len(top_matches) - bullish_count
        
        # Calculate prediction consensus strength
        future_changes = [m['future_change'] for m in top_matches]
        prediction_std = np.std(future_changes)
        consensus_strength = max(0, 100 - prediction_std * 2)  # Lower std = higher consensus
        
        # Enhanced confidence calculation
        confidence_score = (
            avg_similarity * 0.4 +  # Average similarity weight
            min_similarity * 0.2 +  # Weakest link weight
            consensus_strength * 0.3 +  # Prediction agreement weight
            (max(bullish_count, bearish_count) / len(top_matches)) * 100 * 0.1  # Directional consensus weight
        )
        
        col_pred1, col_pred2, col_pred3, col_pred4 = st.columns(4)
        with col_pred1:
            st.metric("Predicted Change", f"{ensemble_future_change:+.2f}%")
        with col_pred2:
            st.metric("Avg Similarity", f"{avg_similarity:.1f}%", f"Range: {min_similarity:.1f}%-{max([m['similarity'] for m in top_matches]):.1f}%")
        with col_pred3:
            st.metric("Signal Consensus", f"{bullish_count} vs {bearish_count}", f"{max(bullish_count, bearish_count)/len(top_matches)*100:.0f}% agreement")
        with col_pred4:
            confidence_level = "High" if confidence_score >= 75 else "Medium" if confidence_score >= 60 else "Low"
            st.metric("Confidence", f"{confidence_score:.1f}%", confidence_level)
        
        
        # Generate future timeline matching historical data spacing
        last_time = reference_data['Time'].iloc[-1]
        
        # Calculate the most common time interval from reference data
        if len(reference_data) >= 2:
            ref_time_diffs = reference_data['Time'].diff().dropna()
            if len(ref_time_diffs) > 0:
                # Use the most frequent time difference
                time_freq = ref_time_diffs.mode().iloc[0] if len(ref_time_diffs.mode()) > 0 else ref_time_diffs.median()
            else:
                time_freq = pd.Timedelta(hours=1)  # fallback
        else:
            time_freq = pd.Timedelta(hours=1)  # fallback
            
        # Show the detected frequency for debugging
        st.info(f"Detected time interval: {time_freq} for consistent chart display")
        
        # Generate future timestamps matching historical spacing
        future_times = []
        current_time = last_time
        for i in range(20):
            current_time += time_freq
            future_times.append(current_time)
        
        # Create prediction dataframe with proper data types
        pred_df = pd.DataFrame(
            prediction_candles, 
            columns=['Open', 'High', 'Low', 'Close']
        ).astype(np.float64)
        pred_df['Time'] = future_times
        
        # Validate prediction data integrity
        pred_df = pred_df.replace([np.inf, -np.inf], np.nan).dropna()
        if len(pred_df) == 0:
            st.error("Unable to generate valid predictions. Try different parameters.")
            st.stop()
            
        # Simple validation - just ensure we have data
        if len(pred_df) < 20:
            st.warning(f"Generated {len(pred_df)} prediction candles instead of 20. This might affect accuracy.")
            
        # Apply prediction smoothing to reduce noise (NEW FEATURE)
        if len(pred_df) > 2:
            # Apply a gentle smoothing to make predictions more realistic
            window_size = min(3, len(pred_df))
            for col in ['Open', 'High', 'Low', 'Close']:
                pred_df[col] = pred_df[col].rolling(window=window_size, center=True, min_periods=1).mean()
            
            # Ensure OHLC relationships are maintained after smoothing
            for i in range(len(pred_df)):
                open_val = pred_df.iloc[i]['Open']
                high_val = pred_df.iloc[i]['High'] 
                low_val = pred_df.iloc[i]['Low']
                close_val = pred_df.iloc[i]['Close']
                
                # Fix any OHLC violations
                corrected_high = max(high_val, open_val, close_val)
                corrected_low = min(low_val, open_val, close_val)
                
                pred_df.iloc[i, pred_df.columns.get_loc('High')] = corrected_high
                pred_df.iloc[i, pred_df.columns.get_loc('Low')] = corrected_low
        
        # Display prediction chart
        st.markdown("### Price Prediction Chart")
        
        fig_prediction = go.Figure()
        
        # Add historical reference pattern
        fig_prediction.add_trace(go.Candlestick(
            x=reference_data['Time'],
            open=reference_data['Open'],
            high=reference_data['High'],
            low=reference_data['Low'],
            close=reference_data['Close'],
            increasing=dict(
                line=dict(color='#2E8B57', width=3),
                fillcolor='rgba(46, 139, 87, 0.7)'
            ),
            decreasing=dict(
                line=dict(color='#CD5C5C', width=3),
                fillcolor='rgba(205, 92, 92, 0.7)'
            ),
            name="Historical (20 candles)"
        ))
        
        # Add prediction with adaptive candlestick sizing
        fig_prediction.add_trace(go.Candlestick(
            x=pred_df['Time'],
            open=pred_df['Open'],
            high=pred_df['High'],
            low=pred_df['Low'],
            close=pred_df['Close'],
            increasing=dict(
                line=dict(color='#00FF00', width=4),
                fillcolor='rgba(0, 255, 0, 0.8)'
            ),
            decreasing=dict(
                line=dict(color='#FF0000', width=4),
                fillcolor='rgba(255, 0, 0, 0.8)'
            ),
            name="AI Prediction (20 candles)"
        ))
        
        # Add actual future data if available (for comparison)
        if len(future_data) == 20:
            fig_prediction.add_trace(go.Candlestick(
                x=future_data['Time'],
                open=future_data['Open'],
                high=future_data['High'],
                low=future_data['Low'],
                close=future_data['Close'],
                increasing=dict(
                    line=dict(color='#FFD700', width=3),
                    fillcolor='rgba(255, 215, 0, 0.7)'
                ),
                decreasing=dict(
                    line=dict(color='#FFA500', width=3),
                    fillcolor='rgba(255, 165, 0, 0.7)'
                ),
                name="Actual Future (Validation)"
            ))
        
        # Add separator line (with error handling)
        separator_time = reference_data['Time'].iloc[-1]
        
        # Calculate y-axis range with error handling
        try:
            ref_low_min = reference_data['Low'].min()
            ref_high_max = reference_data['High'].max()
            pred_low_min = pred_df['Low'].min() if len(pred_df) > 0 else ref_low_min
            pred_high_max = pred_df['High'].max() if len(pred_df) > 0 else ref_high_max
            
            y_min = min(ref_low_min, pred_low_min)
            y_max = max(ref_high_max, pred_high_max)
            
            # Add some padding
            y_range = y_max - y_min
            y_min -= y_range * 0.05
            y_max += y_range * 0.05
        except:
            # Fallback to reference data range
            y_min = reference_data['Low'].min()
            y_max = reference_data['High'].max()
        
        fig_prediction.add_shape(
            type="line",
            x0=separator_time, y0=y_min,
            x1=separator_time, y1=y_max,
            line=dict(color="#FFD700", width=3, dash="dash")
        )
        
        # Calculate proper x-axis range to avoid huge gaps
        all_times = list(reference_data['Time']) + list(pred_df['Time'])
        if len(future_data) == 20:
            all_times.extend(list(future_data['Time']))
        
        min_time = min(all_times)
        max_time = max(all_times)
        
        # Calculate appropriate tick spacing based on time range
        time_range = max_time - min_time
        
        # Determine optimal number of ticks based on time range
        if time_range <= pd.Timedelta(hours=2):
            nticks = 10
            tick_format = '%H:%M'
        elif time_range <= pd.Timedelta(days=1):
            nticks = 12
            tick_format = '%m-%d %H:%M'
        elif time_range <= pd.Timedelta(days=7):
            nticks = 15
            tick_format = '%m-%d'
        else:
            nticks = 20
            tick_format = '%Y-%m-%d'
            
        fig_prediction.update_layout(
            height=600,
            title="AI Price Prediction: Historical vs Predicted vs Actual",
            xaxis_rangeslider_visible=False,
            showlegend=True,
            margin=dict(l=20, r=20, t=60, b=20),
            xaxis=dict(
                type='date',
                range=[min_time, max_time],
                tickmode='auto',
                nticks=nticks,
                tickformat=tick_format,
                rangeslider=dict(visible=False),
                showgrid=True,
                gridcolor='rgba(128, 128, 128, 0.2)',
                tickangle=-45
            ),
            yaxis=dict(
                autorange=True,
                fixedrange=False,
                showgrid=True,
                gridcolor='rgba(128, 128, 128, 0.2)'
            )
        )
        
        st.plotly_chart(fig_prediction, use_container_width=True)
        
        # Validate prediction accuracy button
        st.markdown("### Prediction Validation")
        
        if len(future_data) == 20:
            col_validate1, col_validate2 = st.columns([1, 3])
            
            with col_validate1:
                if st.button("Validate Prediction Accuracy", type="secondary", use_container_width=True):
                    st.session_state.show_validation = True
            
            with col_validate2:
                st.info("Compare AI prediction against actual market data to measure accuracy")
            
            # Show validation results if button was clicked
            if st.session_state.get('show_validation', False):
                st.markdown("#### 🎯 Prediction Accuracy Results")
                
                # Calculate comprehensive accuracy metrics
                actual_change = ((future_data['Close'].iloc[-1] - future_data['Close'].iloc[0]) / future_data['Close'].iloc[0]) * 100
                prediction_error = abs(ensemble_future_change - actual_change)
                direction_correct = (ensemble_future_change > 0) == (actual_change > 0)
                
                # Price prediction accuracy (inverse of relative error)
                relative_error = prediction_error / (abs(actual_change) + 0.01)  # Add small constant to avoid division by zero
                price_accuracy = max(0, 100 - (relative_error * 10))  # Scale error to percentage
                price_accuracy = min(price_accuracy, 100)  # Cap at 100%
                
                # Direction accuracy
                direction_accuracy = 100 if direction_correct else 0
                
                # Overall accuracy (weighted average)
                overall_accuracy = (price_accuracy * 0.7) + (direction_accuracy * 0.3)
                
                # Calculate candle-by-candle accuracy
                candle_accuracies = []
                for i in range(min(20, len(future_data), len(pred_df))):
                    actual_close = future_data['Close'].iloc[i]
                    predicted_close = pred_df['Close'].iloc[i]
                    candle_error = abs(predicted_close - actual_close) / actual_close * 100
                    candle_accuracy = max(0, 100 - candle_error)
                    candle_accuracies.append(candle_accuracy)
                
                avg_candle_accuracy = np.mean(candle_accuracies) if candle_accuracies else 0
                
                # Display accuracy metrics with color coding
                col_metric1, col_metric2, col_metric3, col_metric4 = st.columns(4)
                
                with col_metric1:
                    st.metric(
                        "Overall Accuracy",
                        f"{overall_accuracy:.1f}%",
                        delta=f"{overall_accuracy - 50:.1f}% vs baseline" if overall_accuracy > 50 else None
                    )
                
                with col_metric2:
                    st.metric(
                        "Direction Accuracy",
                        "100%" if direction_correct else "0%",
                        delta="Correct" if direction_correct else "Wrong"
                    )
                
                with col_metric3:
                    st.metric(
                        "Price Accuracy",
                        f"{price_accuracy:.1f}%",
                        delta=f"Error: {prediction_error:.2f}%"
                    )
                
                with col_metric4:
                    st.metric(
                        "Avg Candle Accuracy",
                        f"{avg_candle_accuracy:.1f}%",
                        delta=f"{len(candle_accuracies)} candles analyzed"
                    )
                
                # Detailed comparison
                st.markdown("#### Detailed Prediction vs Actual")
                
                comparison_data = {
                    "Metric": ["Predicted Change", "Actual Change", "Error", "Direction Match"],
                    "Value": [
                        f"{ensemble_future_change:+.2f}%",
                        f"{actual_change:+.2f}%",
                        f"{prediction_error:.2f}%",
                        "Yes" if direction_correct else "No"
                    ]
                }
                
                st.table(pd.DataFrame(comparison_data))
                
                # Interpretation
                st.markdown("#### Accuracy Interpretation")
                if overall_accuracy >= 80:
                    st.success("**Excellent Prediction**: The AI model achieved high accuracy with strong price and direction predictions.")
                elif overall_accuracy >= 60:
                    st.warning("**Good Prediction**: Reasonable accuracy with some deviation from actual prices.")
                elif overall_accuracy >= 40:
                    st.info("**Moderate Prediction**: Mixed results - some aspects predicted correctly, others less so.")
                else:
                    st.error("**Poor Prediction**: Low accuracy. The selected patterns may not have been truly similar.")
                
                # Reset validation state
                if st.button("Clear Validation", key="clear_validation"):
                    st.session_state.show_validation = False
                    st.rerun()
                    
        else:
            st.warning("No future data available for validation. Select an earlier prediction date to enable accuracy testing.")
        
        # Show confidence metrics if requested
        if show_confidence:
            # Additional confidence metrics
            pattern_consistency = np.std([m['future_change'] for m in top_matches])
            similarity_spread = max([m['similarity'] for m in top_matches]) - min([m['similarity'] for m in top_matches])
            
            col_conf1, col_conf2, col_conf3 = st.columns(3)
            with col_conf1:
                st.metric("Pattern Consistency", f"{pattern_consistency:.2f}%", help="Lower values indicate more consistent predictions")
            with col_conf2:
                st.metric("Similarity Spread", f"{similarity_spread:.1f}%", help="Lower spread indicates more uniform match quality")
            with col_conf3:
                confidence_score = max(0, 100 - pattern_consistency * 2 - similarity_spread * 0.5)
                confidence_level = "High" if confidence_score >= 70 else "Medium" if confidence_score >= 50 else "Low"
                st.metric("Confidence Score", f"{confidence_score:.1f}% ({confidence_level})")
        
        
        # Show individual pattern matches if requested
        if show_individual_predictions:
            st.markdown(f"### Top {len(top_matches)} Pattern Matches")
            
            for i, match in enumerate(top_matches):
                with st.expander(
                    f"Match #{i+1} - {match['start_time'].strftime('%Y-%m-%d %H:%M')} | "
                    f"Similarity: {match['similarity']}% | Predicted: {match['future_change']:+.1f}%",
                    expanded=i < 2
                ):
                    # Match details
                    col_detail1, col_detail2, col_detail3 = st.columns(3)
                    with col_detail1:
                        st.metric("Pattern Similarity", f"{match['similarity']}%")
                    with col_detail2:
                        st.metric("Pattern Change", f"{match['pattern_change']:+.1f}%")
                    with col_detail3:
                        st.metric("Future Prediction", f"{match['future_change']:+.1f}%")
                    
                    # Combined chart showing historical pattern + its actual future
                    fig_match = go.Figure()
                    
                    # Historical pattern
                    fig_match.add_trace(go.Candlestick(
                        x=match['pattern_data']['Time'],
                        open=match['pattern_data']['Open'],
                        high=match['pattern_data']['High'],
                        low=match['pattern_data']['Low'],
                        close=match['pattern_data']['Close'],
                        increasing=dict(
                            line=dict(color='#00AA00', width=3),
                            fillcolor='rgba(0, 170, 0, 0.7)'
                        ),
                        decreasing=dict(
                            line=dict(color='#FF0000', width=3),
                            fillcolor='rgba(255, 0, 0, 0.7)'
                        ),
                        name="Historical Pattern"
                    ))
                    
                    # Its actual future
                    fig_match.add_trace(go.Candlestick(
                        x=match['future_data']['Time'],
                        open=match['future_data']['Open'],
                        high=match['future_data']['High'],
                        low=match['future_data']['Low'],
                        close=match['future_data']['Close'],
                        increasing=dict(
                            line=dict(color='#FFD700', width=3),
                            fillcolor='rgba(255, 215, 0, 0.7)'
                        ),
                        decreasing=dict(
                            line=dict(color='#FFA500', width=3),
                            fillcolor='rgba(255, 165, 0, 0.7)'
                        ),
                        name="Actual Outcome"
                    ))
                    
                    fig_match.update_layout(
                        height=300,
                        title=f"Historical Match {i+1}: Pattern → Actual Outcome",
                        xaxis_rangeslider_visible=False,
                        margin=dict(l=20, r=20, t=40, b=20),
                        showlegend=True,
                        xaxis=dict(
                            type='date',
                            showgrid=True,
                            gridcolor='rgba(128, 128, 128, 0.2)'
                        ),
                        yaxis=dict(
                            showgrid=True,
                            gridcolor='rgba(128, 128, 128, 0.2)'
                        )
                    )
                    st.plotly_chart(fig_match, use_container_width=True)

