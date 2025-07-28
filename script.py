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


tab1, tab2, tab3 , tab4 = st.tabs(["Trading View", "Compare Sequences", "Pattern Finder","Auto Prediction"])


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
    st.subheader("🔮 AI Price Prediction — Pattern-Based Forecasting")
    
    # Check if data is available
    if df.empty:
        st.error("❌ No data available for predictions. Please check your data file.")
        st.stop()
    
    # Enhanced UI layout
    st.markdown("### 📊 Prediction Configuration")
    
    col1, col2, col3 = st.columns([2, 2, 1])
    
    with col1:
        st.markdown("**📅 Reference Point**")
        # Default to a realistic date from the data
        default_date = df['Time'].max().date() - timedelta(days=1)
        date_ref = st.date_input("Reference Date", value=default_date)
        time_ref = st.time_input("Reference Time", value=datetime.strptime("14:00", "%H:%M").time())
        
    with col2:
        st.markdown("**⚙️ Prediction Settings**")
        window_len = st.slider("Historical Pattern Length", 10, 50, 20, help="Number of candles to analyze for pattern matching")
        future_candles_to_show = st.slider("Prediction Horizon", 5, 30, 15, help="Number of future candles to predict")
        
    with col3:
        st.markdown("**🎯 Quality Control**")
        min_correlation = st.slider("Min Pattern Similarity", 30, 90, 60, help="Minimum correlation % to consider a pattern match")
        max_predictions = st.number_input("Max Predictions", 3, 10, 5)
    
    # Combine date and time
    ref_datetime = datetime.combine(date_ref, time_ref)
    
    # Generate prediction button
    if st.button("🚀 Generate AI Predictions", type="primary", use_container_width=True):
        
        # Validation
        pattern_df = df[df['Time'] <= ref_datetime].tail(window_len).copy()
        if len(pattern_df) < window_len:
            st.error(f"❌ Insufficient data. Found {len(pattern_df)} candles, need {window_len}. Try selecting an earlier reference date.")
            st.stop()
        
        # Show reference pattern
        st.markdown("### 📈 Reference Pattern Analysis")
        
        # Calculate pattern metrics
        price_change = ((pattern_df['Close'].iloc[-1] - pattern_df['Close'].iloc[0]) / pattern_df['Close'].iloc[0]) * 100
        volatility = (pattern_df['High'].max() - pattern_df['Low'].min()) / pattern_df['Close'].mean() * 100
        avg_volume = (pattern_df['High'] - pattern_df['Low']).mean()
        
        col_metrics1, col_metrics2, col_metrics3, col_metrics4 = st.columns(4)
        with col_metrics1:
            st.metric("Pattern Length", f"{window_len} candles")
        with col_metrics2:
            st.metric("Price Change", f"{price_change:+.2f}%")
        with col_metrics3:
            st.metric("Volatility", f"{volatility:.1f}%")
        with col_metrics4:
            st.metric("Avg Range", f"{avg_volume:.0f}")
        
        # Display reference pattern chart
        fig_ref = go.Figure()
        fig_ref.add_trace(go.Candlestick(
            x=pattern_df['Time'],
            open=pattern_df['Open'],
            high=pattern_df['High'],
            low=pattern_df['Low'],
            close=pattern_df['Close'],
            increasing_line_color='green',
            decreasing_line_color='red',
            name="Reference Pattern"
        ))
        fig_ref.update_layout(
            height=400,
            title=f"Reference Pattern: {pattern_df['Time'].iloc[0].strftime('%Y-%m-%d %H:%M')} to {pattern_df['Time'].iloc[-1].strftime('%Y-%m-%d %H:%M')}",
            xaxis_rangeslider_visible=False,
            margin=dict(l=10, r=10, t=50, b=10)
        )
        st.plotly_chart(fig_ref, use_container_width=True)
        
        # Advanced pattern matching with caching
        @st.cache_data(ttl=120)
        def find_similar_patterns_for_prediction(ref_pattern_hash, window_length, future_length, min_corr_threshold):
            """Find historical patterns similar to reference pattern"""
            ref_data = st.session_state.df_data[st.session_state.df_data['Time'] <= ref_datetime].tail(window_length)
            
            # Normalize reference pattern
            ref_close = zscore(ref_data['Close'].values.astype(np.float32))
            ref_high = zscore(ref_data['High'].values.astype(np.float32))
            ref_low = zscore(ref_data['Low'].values.astype(np.float32))
            
            matches = []
            total_windows = len(df) - window_length - future_length
            step_size = max(3, total_windows // 500)  # Limit to 500 comparisons for speed
            
            # Progress tracking
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i in range(0, total_windows, step_size):
                if i % 50 == 0:
                    progress_bar.progress(min(i / total_windows, 1.0))
                    status_text.text(f"Scanning historical patterns... {i}/{total_windows}")
                
                candidate_df = df.iloc[i:i + window_length]
                future_df = df.iloc[i + window_length:i + window_length + future_length]
                
                # Skip if overlaps with reference or insufficient future data
                if (len(candidate_df) < window_length or len(future_df) < future_length or
                    candidate_df['Time'].iloc[-1] >= ref_datetime):
                    continue
                
                try:
                    # Calculate multi-feature similarity
                    cand_close = zscore(candidate_df['Close'].values.astype(np.float32))
                    cand_high = zscore(candidate_df['High'].values.astype(np.float32))
                    cand_low = zscore(candidate_df['Low'].values.astype(np.float32))
                    
                    # Weighted correlation (Close price has more weight)
                    corr_close = np.corrcoef(ref_close, cand_close)[0, 1]
                    corr_high = np.corrcoef(ref_high, cand_high)[0, 1]
                    corr_low = np.corrcoef(ref_low, cand_low)[0, 1]
                    
                    if np.isnan(corr_close) or np.isnan(corr_high) or np.isnan(corr_low):
                        continue
                    
                    # Combined similarity score
                    similarity = (0.6 * corr_close + 0.2 * corr_high + 0.2 * corr_low) * 100
                    
                    if similarity >= min_corr_threshold:
                        # Calculate pattern metrics
                        pattern_change = ((candidate_df['Close'].iloc[-1] - candidate_df['Close'].iloc[0]) / 
                                        candidate_df['Close'].iloc[0]) * 100
                        future_change = ((future_df['Close'].iloc[-1] - candidate_df['Close'].iloc[-1]) / 
                                       candidate_df['Close'].iloc[-1]) * 100
                        
                        matches.append({
                            "start": candidate_df['Time'].iloc[0],
                            "end": candidate_df['Time'].iloc[-1],
                            "similarity": round(similarity, 2),
                            "pattern_change": round(pattern_change, 2),
                            "future_change": round(future_change, 2),
                            "pattern_data": candidate_df.copy(),
                            "future_data": future_df.copy()
                        })
                        
                except (ValueError, IndexError, ZeroDivisionError):
                    continue
            
            # Clean up progress indicators
            progress_bar.empty()
            status_text.empty()
            
            return sorted(matches, key=lambda x: x['similarity'], reverse=True)
        
        # Execute pattern search
        with st.spinner(f"🔍 Searching for similar patterns... This may take a moment..."):
            similar_matches = find_similar_patterns_for_prediction(
                hash(pattern_df.values.tobytes()), window_len, future_candles_to_show, min_correlation
            )
        
        # Display results
        if not similar_matches:
            st.warning(f"❌ No similar patterns found with similarity >= {min_correlation}%. Try lowering the minimum similarity threshold.")
            st.info("💡 **Suggestions**: Lower the minimum similarity, increase pattern length, or try a different reference point.")
        else:
            st.success(f"✅ Found {len(similar_matches)} similar historical patterns!")
            
            # Show prediction summary
            top_matches = similar_matches[:max_predictions]
            avg_future_change = np.mean([m['future_change'] for m in top_matches])
            bullish_predictions = sum(1 for m in top_matches if m['future_change'] > 0)
            bearish_predictions = len(top_matches) - bullish_predictions
            
            st.markdown("### 🎯 AI Prediction Summary")
            col_sum1, col_sum2, col_sum3, col_sum4 = st.columns(4)
            with col_sum1:
                st.metric("Avg Future Change", f"{avg_future_change:+.1f}%")
            with col_sum2:
                st.metric("Bullish Signals", f"{bullish_predictions}/{len(top_matches)}")
            with col_sum3:
                st.metric("Bearish Signals", f"{bearish_predictions}/{len(top_matches)}")
            with col_sum4:
                confidence = "High" if len(top_matches) >= 3 and top_matches[0]['similarity'] >= 75 else "Medium" if len(top_matches) >= 2 else "Low"
                st.metric("Confidence", confidence)
            
            # Generate future predictions
            st.markdown("### 🔮 Price Predictions")
            
            # Calculate prediction based on historical patterns
            last_price = pattern_df['Close'].iloc[-1]
            last_time = pattern_df['Time'].iloc[-1]
            time_freq = df['Time'].diff().mode()[0]
            
            # Create prediction timeline
            future_times = [last_time + (i+1) * time_freq for i in range(future_candles_to_show)]
            
            # Aggregate predictions from top matches
            prediction_values = np.zeros((future_candles_to_show, 4), dtype=np.float64)  # OHLC
            
            for match in top_matches:
                # Weight by similarity
                weight = float(match['similarity']) / 100.0
                
                # Calculate deltas from historical pattern - ensure float64
                base_values = match['pattern_data'].iloc[-1][['Open', 'High', 'Low', 'Close']].values.astype(np.float64)
                future_values = match['future_data'][['Open', 'High', 'Low', 'Close']].values.astype(np.float64)
                current_values = pattern_df.iloc[-1][['Open', 'High', 'Low', 'Close']].values.astype(np.float64)
                
                # Apply deltas to current price level
                for j in range(min(len(future_values), future_candles_to_show)):
                    # Calculate delta and apply to current values
                    delta = future_values[j] - base_values
                    predicted_candle = current_values + delta
                    prediction_values[j] += predicted_candle * weight
            
            # Normalize by total weight
            total_weight = sum(float(m['similarity']) / 100.0 for m in top_matches)
            if total_weight > 0:
                prediction_values /= total_weight
            
            # Create prediction dataframe
            pred_df = pd.DataFrame(prediction_values, columns=['Open', 'High', 'Low', 'Close'])
            pred_df['Time'] = future_times[:len(pred_df)]
            
            # Display prediction chart
            fig_pred = go.Figure()
            
            # Add historical pattern (reference)
            fig_pred.add_trace(go.Candlestick(
                x=pattern_df['Time'],
                open=pattern_df['Open'],
                high=pattern_df['High'],
                low=pattern_df['Low'],
                close=pattern_df['Close'],
                increasing_line_color='green',
                decreasing_line_color='red',
                name="Historical Reference",
                opacity=0.7
            ))
            
            # Add prediction
            fig_pred.add_trace(go.Candlestick(
                x=pred_df['Time'],
                open=pred_df['Open'],
                high=pred_df['High'],
                low=pred_df['Low'],
                close=pred_df['Close'],
                increasing_line_color='green',
                decreasing_line_color='red',
                name="AI Prediction"
            ))
            
            fig_pred.update_layout(
                height=500,
                title=f"AI Price Prediction - Next {future_candles_to_show} Candles",
                xaxis_rangeslider_visible=False,
                margin=dict(l=10, r=10, t=50, b=10),
                showlegend=True
            )
            st.plotly_chart(fig_pred, use_container_width=True)
            
            # Show detailed pattern matches
            st.markdown(f"### 📊 Top {len(top_matches)} Historical Pattern Matches")
            
            for i, match in enumerate(top_matches):
                similarity_color = "🟢" if match['similarity'] >= 80 else "🟡" if match['similarity'] >= 70 else "🔵"
                trend_emoji = "📈" if match['future_change'] > 0 else "📉"
                
                with st.expander(
                    f"{similarity_color} {trend_emoji} Match #{i+1} — {match['start'].strftime('%Y-%m-%d %H:%M')} | "
                    f"Similarity: {match['similarity']}% | Future: {match['future_change']:+.1f}%",
                    expanded=i < 2  # Auto-expand top 2
                ):
                    # Pattern details
                    col_detail1, col_detail2, col_detail3 = st.columns(3)
                    with col_detail1:
                        st.metric("Pattern Similarity", f"{match['similarity']}%")
                    with col_detail2:
                        st.metric("Pattern Change", f"{match['pattern_change']:+.1f}%")
                    with col_detail3:
                        st.metric("Future Outcome", f"{match['future_change']:+.1f}%")
                    
                    # Combined chart showing pattern + future
                    combined_data = pd.concat([match['pattern_data'], match['future_data']], ignore_index=True)
                    
                    fig_match = go.Figure()
                    
                    # Historical pattern
                    fig_match.add_trace(go.Candlestick(
                        x=match['pattern_data']['Time'],
                        open=match['pattern_data']['Open'],
                        high=match['pattern_data']['High'],
                        low=match['pattern_data']['Low'],
                        close=match['pattern_data']['Close'],
                        increasing_line_color='green',
                        decreasing_line_color='red',
                        name="Historical Pattern"
                    ))
                    
                    # Future outcome
                    fig_match.add_trace(go.Candlestick(
                        x=match['future_data']['Time'],
                        open=match['future_data']['Open'],
                        high=match['future_data']['High'],
                        low=match['future_data']['Low'],
                        close=match['future_data']['Close'],
                        increasing_line_color='green',
                        decreasing_line_color='red',
                        name="Actual Future"
                    ))
                    
                    fig_match.update_layout(
                        height=350,
                        title=f"Historical Match {i+1} + Actual Outcome",
                        xaxis_rangeslider_visible=False,
                        margin=dict(l=10, r=10, t=40, b=10),
                        showlegend=True
                    )
                    st.plotly_chart(fig_match, use_container_width=True)
            
            # Prediction accuracy assessment
            if st.button("🧪 Test Prediction Accuracy", key="test_accuracy"):
                with st.spinner("Testing prediction accuracy on historical data..."):
                    # Quick accuracy test on recent data
                    test_results = []
                    test_sample_size = 20
                    
                    recent_dates = df['Time'].tail(test_sample_size * (window_len + future_candles_to_show))
                    test_points = recent_dates[::window_len + future_candles_to_show][:test_sample_size]
                    
                    correct_predictions = 0
                    total_predictions = 0
                    
                    for test_point in test_points:
                        test_pattern = df[df['Time'] <= test_point].tail(window_len)
                        actual_future = df[(df['Time'] > test_point) & 
                                         (df['Time'] <= test_point + timedelta(minutes=5*future_candles_to_show))]
                        
                        if len(test_pattern) == window_len and len(actual_future) >= future_candles_to_show:
                            # Simple prediction: use average of top matches
                            predicted_direction = 1 if avg_future_change > 0 else -1
                            actual_change = ((actual_future['Close'].iloc[future_candles_to_show-1] - test_pattern['Close'].iloc[-1]) / 
                                           test_pattern['Close'].iloc[-1]) * 100
                            actual_direction = 1 if actual_change > 0 else -1
                            
                            if predicted_direction == actual_direction:
                                correct_predictions += 1
                            total_predictions += 1
                    
                    if total_predictions > 0:
                        accuracy = (correct_predictions / total_predictions) * 100
                        st.success(f"📊 **Historical Accuracy Test**: {accuracy:.1f}% ({correct_predictions}/{total_predictions} predictions correct)")
                        
                        if accuracy >= 70:
                            st.success("🎯 High confidence in prediction model!")
                        elif accuracy >= 55:
                            st.warning("⚠️ Moderate confidence - use with caution")
                        else:
                            st.error("🚨 Low confidence - consider different parameters")
                    else:
                        st.warning("Insufficient data for accuracy testing")

