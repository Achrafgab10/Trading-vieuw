import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import date, datetime, timedelta
import random
import warnings

warnings.filterwarnings('ignore')

# Simple page config
st.set_page_config(
    page_title="BEOM Trading Tools",
    layout="wide", 
    initial_sidebar_state="collapsed"
)

# Clean header
st.title("BEOM Trading Tools")
st.markdown("Trading Analysis Platform")

# Data loading function
@st.cache_data(ttl=300)
def load_data():
    try:
        df = pd.read_csv("data_cleaned.csv", sep=';')
        df['Time'] = pd.to_datetime(df['Time'], format='%d.%m.%Y %H:%M', errors='coerce')
        df = df.dropna().sort_values('Time').reset_index(drop=True)
        return df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return pd.DataFrame(columns=['Time', 'Open', 'High', 'Low', 'Close'])

# Z-score function
def zscore(x):
    x = np.asarray(x, dtype=np.float32)
    std_val = np.std(x)
    return (x - np.mean(x)) / std_val if std_val != 0 else x

# Load data
if 'df_data' not in st.session_state:
    with st.spinner('Loading data...'):
        st.session_state.df_data = load_data()

df = st.session_state.df_data

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["Trading View", "Compare Sequences", "Pattern Finder", "AI Prediction"])

# TAB 1: Trading View (Cleaned)
with tab1:
    st.subheader("Trading View")
    
    col1, col2 = st.columns(2)
    
    with col1:
        start_date = st.date_input("From", df['Time'].min().date())
        end_date = st.date_input("To", df['Time'].max().date())
    
    with col2:
        interval = st.selectbox("Interval", ['5min', '15min', '1h', '4h', '1d'])
    
    # Filter data
    mask = (df['Time'].dt.date >= start_date) & (df['Time'].dt.date <= end_date)
    df_filtered = df.loc[mask]
    
    # Resample data
    rule_map = {'5min': '5min', '15min': '15min', '1h': '1H', '4h': '4H', '1d': '1D'}
    df_resampled = df_filtered.set_index('Time').resample(rule_map[interval]).agg({
        'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last'
    }).dropna().reset_index()
    
    # Create chart
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
        title="Price Chart",
        xaxis_title="Time",
        yaxis_title="Price",
        xaxis_rangeslider_visible=False
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Basic statistics
    if len(df_resampled) > 0:
        st.markdown("### Statistics")
        
        first_price = df_resampled['Close'].iloc[0]
        last_price = df_resampled['Close'].iloc[-1]
        price_change = last_price - first_price
        price_change_pct = (price_change / first_price) * 100
        high_price = df_resampled['High'].max()
        low_price = df_resampled['Low'].min()
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Price Change", f"{price_change:+.2f}", f"{price_change_pct:+.2f}%")
        with col2:
            st.metric("Period High", f"{high_price:.2f}")
        with col3:
            st.metric("Period Low", f"{low_price:.2f}")
        with col4:
            st.metric("Range", f"{high_price - low_price:.2f}")

# TAB 2: Compare Sequences (Simplified)
with tab2:
    st.subheader("Compare Two Trading Sequences")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Sequence 1**")
        date1_start = st.date_input("Start Date 1", date(2024, 1, 15))
        time1_start = st.time_input("Start Time 1", datetime.strptime("10:15", "%H:%M").time())
        date1_end = st.date_input("End Date 1", date(2024, 1, 15))
        time1_end = st.time_input("End Time 1", datetime.strptime("10:55", "%H:%M").time())
        
    with col2:
        st.markdown("**Sequence 2**")
        date2_start = st.date_input("Start Date 2", date(2024, 1, 16))
        time2_start = st.time_input("Start Time 2", datetime.strptime("00:20", "%H:%M").time())
        date2_end = st.date_input("End Date 2", date(2024, 1, 16))
        time2_end = st.time_input("End Time 2", datetime.strptime("01:00", "%H:%M").time())
    
    comparison_method = st.selectbox(
        "Comparison Method:",
        ["Price Correlation", "Movement Direction", "Normalized Shape"]
    )
    
    start1 = datetime.combine(date1_start, time1_start)
    end1 = datetime.combine(date1_end, time1_end)
    start2 = datetime.combine(date2_start, time2_start)
    end2 = datetime.combine(date2_end, time2_end)
    
    if st.button("Compare Sequences"):
        seq1 = df[(df['Time'] >= start1) & (df['Time'] <= end1)].copy()
        seq2 = df[(df['Time'] >= start2) & (df['Time'] <= end2)].copy()
        
        if len(seq1) == 0 or len(seq2) == 0:
            st.error("No data found for selected time ranges")
        else:
            # Handle different lengths
            if len(seq1) != len(seq2):
                min_len = min(len(seq1), len(seq2))
                seq1 = seq1.head(min_len)
                seq2 = seq2.head(min_len)
                st.info(f"Trimmed both sequences to {min_len} points")
            
            # Perform comparison
            if comparison_method == "Price Correlation":
                correlation = np.corrcoef(seq1['Close'].values, seq2['Close'].values)[0, 1]
                correlation_pct = correlation * 100 if not np.isnan(correlation) else 0
                st.metric("Correlation", f"{correlation_pct:.2f}%")
                
            elif comparison_method == "Movement Direction":
                moves1 = np.sign(np.diff(seq1['Close'].values))
                moves2 = np.sign(np.diff(seq2['Close'].values))
                direction_match = (moves1 == moves2).mean() * 100
                st.metric("Direction Match", f"{direction_match:.2f}%")
                
            elif comparison_method == "Normalized Shape":
                norm1 = zscore(seq1['Close'].values)
                norm2 = zscore(seq2['Close'].values)
                shape_corr = np.corrcoef(norm1, norm2)[0, 1]
                shape_corr_pct = shape_corr * 100 if not np.isnan(shape_corr) else 0
                st.metric("Shape Similarity", f"{shape_corr_pct:.2f}%")
            
            # Display charts
            col1, col2 = st.columns(2)
            
            with col1:
                fig1 = go.Figure()
                fig1.add_trace(go.Candlestick(
                    x=seq1['Time'], open=seq1['Open'], high=seq1['High'],
                    low=seq1['Low'], close=seq1['Close']
                ))
                fig1.update_layout(height=400, title="Sequence 1", xaxis_rangeslider_visible=False)
                st.plotly_chart(fig1, use_container_width=True)
            
            with col2:
                fig2 = go.Figure()
                fig2.add_trace(go.Candlestick(
                    x=seq2['Time'], open=seq2['Open'], high=seq2['High'],
                    low=seq2['Low'], close=seq2['Close']
                ))
                fig2.update_layout(height=400, title="Sequence 2", xaxis_rangeslider_visible=False)
                st.plotly_chart(fig2, use_container_width=True)

# TAB 3: Pattern Finder (Simplified)
with tab3:
    st.subheader("Pattern Finder")
    
    col1, col2 = st.columns(2)
    
    with col1:
        pattern_start_date = st.date_input("Pattern Start Date", datetime(2024, 1, 15).date())
        pattern_start_time = st.time_input("Pattern Start Time", datetime.strptime("10:00", "%H:%M").time())
        n_candles = st.number_input("Pattern Length", min_value=5, max_value=50, value=15)
        
    with col2:
        search_method = st.selectbox("Method", ["Shape Correlation", "Price Correlation"])
        top_n_results = st.number_input("Top Results", min_value=5, max_value=20, value=10)
        min_correlation = st.slider("Min Similarity %", 10, 95, 60)
    
    pattern_start_ts = datetime.combine(pattern_start_date, pattern_start_time)
    pattern_df = df[df['Time'] >= pattern_start_ts].head(n_candles).copy()
    
    if len(pattern_df) < n_candles:
        st.error(f"Not enough data. Found {len(pattern_df)} candles, need {n_candles}")
    else:
        # Display reference pattern
        fig_pattern = go.Figure()
        fig_pattern.add_trace(go.Candlestick(
            x=pattern_df['Time'], 
            open=pattern_df['Open'], 
            high=pattern_df['High'], 
            low=pattern_df['Low'], 
            close=pattern_df['Close'],
            increasing_line_color='green',
            decreasing_line_color='red',
            increasing_fillcolor='rgba(0, 255, 0, 0.3)',
            decreasing_fillcolor='rgba(255, 0, 0, 0.3)'
        ))
        fig_pattern.update_layout(
            height=300, 
            title="Reference Pattern", 
            xaxis_rangeslider_visible=False,
            margin=dict(l=10, r=10, t=40, b=10)
        )
        st.plotly_chart(fig_pattern, use_container_width=True)
        
        if st.button("Find Similar Patterns"):
            with st.spinner("Searching for similar patterns..."):
                results = []
                
                for i in range(0, len(df) - n_candles, 5):
                    window = df.iloc[i:i + n_candles]
                    if len(window) < n_candles or window['Time'].iloc[0] == pattern_start_ts:
                        continue
                    
                    try:
                        if search_method == "Shape Correlation":
                            ref_norm = zscore(pattern_df['Close'].values)
                            cand_norm = zscore(window['Close'].values)
                            correlation = np.corrcoef(ref_norm, cand_norm)[0, 1] * 100
                        else:
                            correlation = np.corrcoef(pattern_df['Close'].values, window['Close'].values)[0, 1] * 100
                        
                        if not np.isnan(correlation) and correlation >= min_correlation:
                            price_change = ((window['Close'].iloc[-1] - window['Close'].iloc[0]) / window['Close'].iloc[0]) * 100
                            results.append({
                                "start": window['Time'].iloc[0],
                                "similarity": round(correlation, 2),
                                "price_change": round(price_change, 2),
                                "data": window.copy()
                            })
                    except:
                        continue
                
                if results:
                    results = sorted(results, key=lambda x: x['similarity'], reverse=True)[:top_n_results]
                    st.success(f"Found {len(results)} similar patterns")
                    
                    for i, pattern in enumerate(results):
                        with st.expander(f"Pattern {i+1} - Similarity: {pattern['similarity']}%"):
                            st.metric("Price Change", f"{pattern['price_change']:+.1f}%")
                            
                            fig_match = go.Figure()
                            fig_match.add_trace(go.Candlestick(
                                x=pattern['data']['Time'], 
                                open=pattern['data']['Open'],
                                high=pattern['data']['High'], 
                                low=pattern['data']['Low'], 
                                close=pattern['data']['Close'],
                                increasing_line_color='green',
                                decreasing_line_color='red',
                                increasing_fillcolor='rgba(0, 255, 0, 0.3)',
                                decreasing_fillcolor='rgba(255, 0, 0, 0.3)'
                            ))
                            fig_match.update_layout(
                                height=250, 
                                xaxis_rangeslider_visible=False,
                                margin=dict(l=10, r=10, t=30, b=10)
                            )
                            st.plotly_chart(fig_match, use_container_width=True)
                else:
                    st.warning("No similar patterns found")

# TAB 4: AI Prediction (Enhanced)
with tab4:
    st.subheader("AI Price Prediction")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        data_start = df['Time'].min().date()
        data_end = df['Time'].max().date()
        default_date = data_start + timedelta(days=int((data_end - data_start).days * 0.7))
        
        prediction_date = st.date_input("Reference Date", value=default_date)
        prediction_time = st.time_input("Reference Time", value=datetime.strptime("10:00", "%H:%M").time())
        
    with col2:
        similarity_method = st.selectbox("Algorithm", [
            "Multi-Feature Ensemble",
            "Shape Correlation", 
            "Price Correlation"
        ])
        min_similarity_threshold = st.slider("Min Similarity %", 40, 95, 65)
        
    with col3:
        pattern_length = st.number_input("Pattern Length (candles)", min_value=10, max_value=50, value=20)
        num_predictions = st.number_input("Number of Similar Patterns", min_value=3, max_value=20, value=5)
        show_individual_patterns = st.checkbox("Show Individual Patterns", True)
    
    prediction_datetime = datetime.combine(prediction_date, prediction_time)
    
    if st.button("Generate Prediction"):
        reference_data = df[df['Time'] < prediction_datetime].tail(pattern_length).copy()
        
        if len(reference_data) < pattern_length:
            st.error(f"Need at least {pattern_length} historical candles")
        else:
            # Get actual future data for validation
            future_data = df[df['Time'] >= prediction_datetime].head(pattern_length).copy()
            
            # Display reference pattern
            st.markdown("### Reference Pattern")
            ref_price_change = ((reference_data['Close'].iloc[-1] - reference_data['Close'].iloc[0]) / reference_data['Close'].iloc[0]) * 100
            st.metric("Reference Price Change", f"{ref_price_change:+.2f}%")
            
            fig_ref = go.Figure()
            fig_ref.add_trace(go.Candlestick(
                x=reference_data['Time'], open=reference_data['Open'],
                high=reference_data['High'], low=reference_data['Low'], 
                close=reference_data['Close']
            ))
            fig_ref.update_layout(height=300, title=f"Last {pattern_length} Candles", xaxis_rangeslider_visible=False)
            st.plotly_chart(fig_ref, use_container_width=True)
            
            # Find similar patterns
            with st.spinner("Finding similar patterns..."):
                matches = []
                
                search_range = len(df) - (pattern_length * 2)
                for i in range(pattern_length, search_range, 3):
                    candidate_pattern = df.iloc[i:i + pattern_length]
                    candidate_future = df.iloc[i + pattern_length:i + (pattern_length * 2)]
                    
                    if len(candidate_pattern) < pattern_length or len(candidate_future) < pattern_length:
                        continue
                    
                    # Skip overlapping periods
                    if (candidate_pattern['Time'].iloc[-1] >= reference_data['Time'].iloc[0] and 
                        candidate_pattern['Time'].iloc[0] <= reference_data['Time'].iloc[-1]):
                        continue
                    
                    try:
                        if similarity_method == "Shape Correlation":
                            ref_norm = zscore(reference_data['Close'].values)
                            cand_norm = zscore(candidate_pattern['Close'].values)
                            similarity = np.corrcoef(ref_norm, cand_norm)[0, 1] * 100
                        elif similarity_method == "Price Correlation":
                            similarity = np.corrcoef(reference_data['Close'].values, candidate_pattern['Close'].values)[0, 1] * 100
                        else:  # Multi-Feature
                            ref_norm = zscore(reference_data['Close'].values)
                            cand_norm = zscore(candidate_pattern['Close'].values)
                            corr1 = np.corrcoef(ref_norm, cand_norm)[0, 1]
                            
                            ref_returns = np.diff(reference_data['Close'].values) / reference_data['Close'].values[:-1]
                            cand_returns = np.diff(candidate_pattern['Close'].values) / candidate_pattern['Close'].values[:-1]
                            corr2 = np.corrcoef(ref_returns, cand_returns)[0, 1]
                            
                            similarity = (abs(corr1) * 0.7 + abs(corr2) * 0.3) * 100
                        
                        if not np.isnan(similarity) and similarity >= min_similarity_threshold:
                            future_change = ((candidate_future['Close'].iloc[-1] - candidate_pattern['Close'].iloc[-1]) / candidate_pattern['Close'].iloc[-1]) * 100
                            matches.append({
                                'similarity': similarity,
                                'future_change': future_change,
                                'pattern_data': candidate_pattern.copy(),
                                'future_data': candidate_future.copy(),
                                'start_time': candidate_pattern['Time'].iloc[0],
                                'end_time': candidate_pattern['Time'].iloc[-1]
                            })
                    except:
                        continue
                
                if len(matches) < num_predictions:
                    st.warning(f"Found only {len(matches)} patterns. Try lowering similarity threshold or reducing number of patterns.")
                    if len(matches) == 0:
                        st.stop()
                
                # Sort and take top N
                top_matches = sorted(matches, key=lambda x: x['similarity'], reverse=True)[:num_predictions]
                
                # Calculate weighted prediction
                total_weight = sum(m['similarity'] / 100.0 for m in top_matches)
                predicted_change = sum(m['future_change'] * (m['similarity'] / 100.0) for m in top_matches) / total_weight
                
                # Generate prediction OHLC data based on weighted average of similar patterns
                last_known_close = reference_data['Close'].iloc[-1]
                prediction_ohlc = []
                
                for candle_idx in range(pattern_length):
                    weighted_open_returns = []
                    weighted_high_returns = []
                    weighted_low_returns = []
                    weighted_close_returns = []
                    total_candle_weight = 0
                    
                    for match in top_matches:
                        if candle_idx < len(match['future_data']):
                            weight = match['similarity'] / 100.0
                            
                            # Get previous close price for return calculation
                            if candle_idx == 0:
                                prev_close = match['pattern_data']['Close'].iloc[-1]
                            else:
                                prev_close = match['future_data']['Close'].iloc[candle_idx - 1]
                            
                            # Calculate returns for OHLC
                            current_candle = match['future_data'].iloc[candle_idx]
                            open_return = (current_candle['Open'] - prev_close) / prev_close
                            high_return = (current_candle['High'] - prev_close) / prev_close
                            low_return = (current_candle['Low'] - prev_close) / prev_close
                            close_return = (current_candle['Close'] - prev_close) / prev_close
                            
                            weighted_open_returns.append(open_return * weight)
                            weighted_high_returns.append(high_return * weight)
                            weighted_low_returns.append(low_return * weight)
                            weighted_close_returns.append(close_return * weight)
                            total_candle_weight += weight
                    
                    if total_candle_weight > 0:
                        # Calculate weighted average returns
                        avg_open_return = sum(weighted_open_returns) / total_candle_weight
                        avg_high_return = sum(weighted_high_returns) / total_candle_weight
                        avg_low_return = sum(weighted_low_returns) / total_candle_weight
                        avg_close_return = sum(weighted_close_returns) / total_candle_weight
                        
                        # Apply returns to get predicted OHLC
                        if candle_idx == 0:
                            base_price = last_known_close
                        else:
                            base_price = prediction_ohlc[-1]['Close']
                        
                        pred_open = base_price * (1 + avg_open_return)
                        pred_high = base_price * (1 + avg_high_return)
                        pred_low = base_price * (1 + avg_low_return)
                        pred_close = base_price * (1 + avg_close_return)
                        
                        # Ensure OHLC logic is maintained
                        pred_high = max(pred_high, pred_open, pred_close)
                        pred_low = min(pred_low, pred_open, pred_close)
                        
                        prediction_ohlc.append({
                            'Open': pred_open,
                            'High': pred_high,
                            'Low': pred_low,
                            'Close': pred_close
                        })
                    else:
                        # Fallback: use last known values
                        last_ohlc = prediction_ohlc[-1] if prediction_ohlc else {
                            'Open': last_known_close, 'High': last_known_close,
                            'Low': last_known_close, 'Close': last_known_close
                        }
                        prediction_ohlc.append(last_ohlc)
                
                # Calculate predicted change from OHLC data
                if prediction_ohlc:
                    predicted_final_price = prediction_ohlc[-1]['Close']
                    predicted_change_ohlc = ((predicted_final_price - last_known_close) / last_known_close) * 100
                else:
                    predicted_change_ohlc = predicted_change
                
                # Create prediction timeline
                last_time = reference_data['Time'].iloc[-1]
                time_freq = df['Time'].diff().mode()[0] if not df['Time'].diff().mode().empty else pd.Timedelta(minutes=5)
                prediction_times = [last_time + (i + 1) * time_freq for i in range(pattern_length)]
                
                # Display results
                st.markdown("### Prediction Results")
                
                # Calculate additional OHLC metrics
                if prediction_ohlc:
                    predicted_high = max([ohlc['High'] for ohlc in prediction_ohlc])
                    predicted_low = min([ohlc['Low'] for ohlc in prediction_ohlc])
                    predicted_range = ((predicted_high - predicted_low) / last_known_close) * 100
                    predicted_final_close = prediction_ohlc[-1]['Close']
                else:
                    predicted_high = predicted_low = predicted_final_close = last_known_close
                    predicted_range = 0
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Predicted Change", f"{predicted_change_ohlc:+.2f}%", f"Final: {predicted_final_close:.2f}")
                with col2:
                    st.metric("Predicted High", f"{predicted_high:.2f}", f"+{((predicted_high - last_known_close) / last_known_close * 100):+.2f}%")
                with col3:
                    st.metric("Predicted Low", f"{predicted_low:.2f}", f"{((predicted_low - last_known_close) / last_known_close * 100):+.2f}%")
                with col4:
                    st.metric("Predicted Range", f"{predicted_range:.2f}%", f"{predicted_high - predicted_low:.2f}")
                
                # Additional info row
                col5, col6, col7, col8 = st.columns(4)
                with col5:
                    avg_similarity = np.mean([m['similarity'] for m in top_matches])
                    st.metric("Avg Similarity", f"{avg_similarity:.1f}%")
                with col6:
                    bullish_count = sum(1 for m in top_matches if m['future_change'] > 0)
                    st.metric("Bullish Signals", f"{bullish_count}/{len(top_matches)}")
                with col7:
                    st.metric("Patterns Used", f"{len(top_matches)}")
                with col8:
                    if prediction_ohlc:
                        bullish_candles = sum(1 for ohlc in prediction_ohlc if ohlc['Close'] > ohlc['Open'])
                        st.metric("Predicted Bullish Candles", f"{bullish_candles}/{len(prediction_ohlc)}")
                    
                # Create prediction chart
                st.markdown("### Prediction Chart")
                
                fig_prediction = go.Figure()
                
                # Add historical candlesticks
                fig_prediction.add_trace(go.Candlestick(
                    x=reference_data['Time'],
                    open=reference_data['Open'],
                    high=reference_data['High'],
                    low=reference_data['Low'],
                    close=reference_data['Close'],
                    increasing_line_color='#00AA00',
                    decreasing_line_color='#FF0000',
                    name='Historical'
                ))
                
                # Add prediction candlesticks
                pred_opens = [ohlc['Open'] for ohlc in prediction_ohlc]
                pred_highs = [ohlc['High'] for ohlc in prediction_ohlc]
                pred_lows = [ohlc['Low'] for ohlc in prediction_ohlc]
                pred_closes = [ohlc['Close'] for ohlc in prediction_ohlc]
                
                fig_prediction.add_trace(go.Candlestick(
                    x=prediction_times,
                    open=pred_opens,
                    high=pred_highs,
                    low=pred_lows,
                    close=pred_closes,
                    increasing_line_color='#0066FF',
                    decreasing_line_color='#FF6600',
                    increasing=dict(line=dict(width=3), fillcolor='rgba(0, 102, 255, 0.7)'),
                    decreasing=dict(line=dict(width=3), fillcolor='rgba(255, 102, 0, 0.7)'),
                    name='AI Prediction'
                ))
                
                # Add actual future data if available for comparison
                if len(future_data) == pattern_length:
                    fig_prediction.add_trace(go.Candlestick(
                        x=future_data['Time'],
                        open=future_data['Open'],
                        high=future_data['High'],
                        low=future_data['Low'],
                        close=future_data['Close'],
                        increasing_line_color='#00DD00',
                        decreasing_line_color='#DD0000',
                        increasing=dict(line=dict(width=2, color='#00DD00'), fillcolor='rgba(0, 221, 0, 0.5)'),
                        decreasing=dict(line=dict(width=2, color='#DD0000'), fillcolor='rgba(221, 0, 0, 0.5)'),
                        name='Actual Future'
                    ))
                
                # Add vertical line to separate historical from prediction
                separator_time = reference_data['Time'].iloc[-1]
                fig_prediction.add_shape(
                    type="line",
                    x0=separator_time, x1=separator_time,
                    y0=0, y1=1,
                    yref="paper",
                    line=dict(color="gray", width=2, dash="dot")
                )
                
                fig_prediction.update_layout(
                    title="AI Price Prediction vs Historical Data",
                    xaxis_title="Time",
                    yaxis_title="Price",
                    height=500,
                    showlegend=True
                )
                
                st.plotly_chart(fig_prediction, use_container_width=True)
                
                # Confidence assessment
                prediction_std = np.std([m['future_change'] for m in top_matches])
                consensus_strength = max(0, 100 - prediction_std * 2)
                
                confidence_score = (avg_similarity * 0.5 + consensus_strength * 0.5)
                confidence_level = "High" if confidence_score >= 75 else "Medium" if confidence_score >= 60 else "Low"
                
                if confidence_score >= 75:
                    st.success(f"High Confidence ({confidence_score:.1f}%)")
                elif confidence_score >= 60:
                    st.warning(f"Medium Confidence ({confidence_score:.1f}%)")
                else:
                    st.error(f"Low Confidence ({confidence_score:.1f}%)")
                
                # Show individual patterns if requested
                if show_individual_patterns:
                    st.markdown(f"### Top {len(top_matches)} Similar Patterns Found")
                    
                    for i, match in enumerate(top_matches):
                        with st.expander(f"Pattern {i+1} - Similarity: {match['similarity']:.1f}% | Future Change: {match['future_change']:+.2f}%"):
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                st.metric("Similarity Score", f"{match['similarity']:.1f}%")
                            with col2:
                                st.metric("Future Change", f"{match['future_change']:+.2f}%")
                            with col3:
                                pattern_change = ((match['pattern_data']['Close'].iloc[-1] - match['pattern_data']['Close'].iloc[0]) / match['pattern_data']['Close'].iloc[0]) * 100
                                st.metric("Pattern Change", f"{pattern_change:+.2f}%")
                            
                            # Show the historical pattern and its actual outcome as candlesticks
                            fig_match = go.Figure()
                            
                            # Historical pattern candlesticks
                            fig_match.add_trace(go.Candlestick(
                                x=match['pattern_data']['Time'],
                                open=match['pattern_data']['Open'],
                                high=match['pattern_data']['High'],
                                low=match['pattern_data']['Low'],
                                close=match['pattern_data']['Close'],
                                increasing_line_color='green',
                                decreasing_line_color='red',
                                increasing_fillcolor='rgba(0, 255, 0, 0.3)',
                                decreasing_fillcolor='rgba(255, 0, 0, 0.3)',
                                name='Historical Pattern'
                            ))
                            
                            # Its actual future outcome candlesticks
                            fig_match.add_trace(go.Candlestick(
                                x=match['future_data']['Time'],
                                open=match['future_data']['Open'],
                                high=match['future_data']['High'],
                                low=match['future_data']['Low'],
                                close=match['future_data']['Close'],
                                increasing_line_color='darkgreen',
                                decreasing_line_color='darkred',
                                increasing_fillcolor='rgba(0, 200, 0, 0.5)',
                                decreasing_fillcolor='rgba(200, 0, 0, 0.5)',
                                name='Actual Outcome'
                            ))
                            
                            # Add separator
                            separator_time_match = match['pattern_data']['Time'].iloc[-1]
                            fig_match.add_shape(
                                type="line",
                                x0=separator_time_match, x1=separator_time_match,
                                y0=0, y1=1,
                                yref="paper",
                                line=dict(color="gray", width=1, dash="dot")
                            )
                            
                            fig_match.update_layout(
                                title=f"Pattern {i+1}: {match['start_time'].strftime('%Y-%m-%d %H:%M')}",
                                xaxis_title="Time",
                                yaxis_title="Price",
                                height=350,
                                showlegend=True,
                                xaxis_rangeslider_visible=False,
                                margin=dict(l=10, r=10, t=40, b=10)
                            )
                            
                            st.plotly_chart(fig_match, use_container_width=True)
                
                # Show validation if future data available
                if len(future_data) == pattern_length:
                    if st.button("Validate Prediction"):
                            actual_change = ((future_data['Close'].iloc[-1] - future_data['Close'].iloc[0]) / future_data['Close'].iloc[0]) * 100
                            prediction_error = abs(predicted_change_ohlc - actual_change)
                            direction_correct = (predicted_change_ohlc > 0) == (actual_change > 0)
                            
                            # Additional OHLC validation metrics
                            actual_high = future_data['High'].max()
                            actual_low = future_data['Low'].min()
                            high_error = abs(((predicted_high - last_known_close) / last_known_close) - ((actual_high - last_known_close) / last_known_close)) * 100
                            low_error = abs(((predicted_low - last_known_close) / last_known_close) - ((actual_low - last_known_close) / last_known_close)) * 100
                            
                            st.markdown("### Validation Results")
                            
                            # Main validation metrics
                            col1, col2, col3, col4 = st.columns(4)
                            
                            with col1:
                                st.metric("Actual Change", f"{actual_change:+.2f}%")
                            with col2:
                                st.metric("Prediction Error", f"{prediction_error:.2f}%")
                            with col3:
                                st.metric("Direction", "Correct" if direction_correct else "Wrong")
                            with col4:
                                overall_accuracy = max(0, 100 - prediction_error)
                                st.metric("Accuracy", f"{overall_accuracy:.1f}%")
                            
                            # OHLC validation metrics
                            st.markdown("#### OHLC Prediction Accuracy")
                            col5, col6, col7, col8 = st.columns(4)
                            
                            with col5:
                                st.metric("Predicted High", f"{predicted_high:.2f}", f"Actual: {actual_high:.2f}")
                            with col6:
                                st.metric("Predicted Low", f"{predicted_low:.2f}", f"Actual: {actual_low:.2f}")
                            with col7:
                                st.metric("High Error", f"{high_error:.2f}%")
                            with col8:
                                st.metric("Low Error", f"{low_error:.2f}%")
