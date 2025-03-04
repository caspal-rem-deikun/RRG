import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import date, timedelta

############################
# Predefined Lists
############################
predefined_lists = {
    "SPDR Sector ETFs": ["XLB", "XLE", "XLF", "XLI", "XLK", "XLP", "XLU", "XLV", "XLRE"],
    "FAANG": ["AAPL", "MSFT", "GOOGL", "AMZN", "META"],
    "Mega Caps": [
        "AAPL",   # Apple
        "MSFT",   # Microsoft
        "GOOGL",  # Alphabet (Google)
        "AMZN",   # Amazon
        "META",   # Meta Platforms
        "TSLA",   # Tesla
        "NVDA",   # Nvidia
        "BRK-B",  # Berkshire Hathaway (Class B shares)
        "UNH",    # UnitedHealth Group
        "XOM"     # Exxon Mobil
    ]
}

############################
# Data Period Mapping
############################
period_mapping = {
    "3 months":  3 * 30,
    "6 months":  6 * 30,
    "1 year":    365,
    "2 years":   365 * 2,
    "5 years":   365 * 5,
    "10 years":  365 * 10
}
# For "All time" we'll use a very early date.

############################
# Utility Functions
############################

def fetch_stock_data(symbols, start, end, interval='1d'):
    """Fetch historical price data from Yahoo Finance for the given symbols."""
    data = {}
    for sym in symbols:
        df = yf.download(sym, start=start, end=end, interval=interval, progress=False)
        if df.empty:
            continue
        if 'Adj Close' in df.columns:
            series = df['Adj Close'].copy()
        elif 'Close' in df.columns:
            series = df['Close'].copy()
        else:
            continue
        series.name = sym.upper()
        data[sym.upper()] = series
    return data

def combine_data_to_df(data_dict, resample_freq=None):
    """Combine a dictionary of {symbol: price_series} into a single DataFrame."""
    if not data_dict:
        return pd.DataFrame()
    df = pd.concat(data_dict.values(), axis=1)
    if resample_freq:
        df = df.resample(resample_freq).last()
    return df

def compute_jdk_indicators(stock_series, ref_series_or_one, short_window=5, long_window=10):
    """
    Compute normalized RS Ratio & raw momentum (growth factor).

    1) Normalize: stock_norm = stock / stock[0]
       If ref_series is real: ref_norm = ref / ref[0], then
           RS = (stock_norm / ref_norm) * 100
       Else if ref_series_or_one == "ONE": RS = stock_norm * 100

    2) Compute rolling averages:
           RS_short = rolling mean over short_window
           RS_long  = rolling mean over long_window
       and then RS_Ratio = (RS_short / RS_long) * 100

    3) Compute raw momentum as the ratio of consecutive RS_Ratio values.
    """
    if not isinstance(stock_series, pd.Series) or stock_series.empty:
        return pd.Series(dtype=float), pd.Series(dtype=float)
    
    df = pd.DataFrame({'stock': stock_series.dropna()})
    if df.empty:
        return pd.Series(dtype=float), pd.Series(dtype=float)
    
    df['stock_norm'] = df['stock'] / df['stock'].iloc[0]
    
    if isinstance(ref_series_or_one, pd.Series):
        ref_aligned = ref_series_or_one.reindex(df.index).dropna()
        if ref_aligned.empty:
            return pd.Series(dtype=float), pd.Series(dtype=float)
        df['ref_norm'] = ref_aligned / ref_aligned.iloc[0]
        df['RS'] = (df['stock_norm'] / df['ref_norm']) * 100.0
    else:
        df['RS'] = df['stock_norm'] * 100.0

    n = len(df)
    if n < long_window:
        long_window = n
        short_window = max(2, n // 2)
    
    df['RS_short'] = df['RS'].rolling(short_window).mean()
    df['RS_long']  = df['RS'].rolling(long_window).mean()
    df['RS_Ratio'] = (df['RS_short'] / df['RS_long']) * 100.0

    df['raw_momentum'] = df['RS_Ratio'] / df['RS_Ratio'].shift(1)
    df = df.dropna()
    return df['RS_Ratio'], df['raw_momentum']

def prepare_rrg_dataframe(symbols, df_prices, reference_input="SPY"):
    """
    Build a DataFrame with columns [Date, Symbol, RS_Ratio, RS_Momentum].

    - If reference != "ONE": compute each stock's RS_Ratio & raw momentum normalized against the reference's raw momentum,
      and force the reference stock to (100,100).
    - If reference == "ONE": use self-normalization and only add the anchor if the user explicitly typed "ONE".
    - Also, only overlapping dates (i.e. rows where all symbols have data) remain in df_prices.
    """
    rrg_rows = []
    ref_input = reference_input.upper()
    
    if ref_input == "ONE":
        common_ref = "ONE"
        ref_raw_mom = None
    else:
        if ref_input not in df_prices.columns:
            common_ref = "ONE"
            ref_raw_mom = None
        else:
            ref_series = df_prices[ref_input].dropna()
            _, ref_raw_mom_series = compute_jdk_indicators(ref_series, ref_series)
            ref_raw_mom = ref_raw_mom_series if not ref_raw_mom_series.empty else None
            common_ref = ref_series

    for sym in symbols:
        sym = sym.upper()
        if sym not in df_prices.columns:
            continue
        
        stock_series = df_prices[sym].dropna()
        if len(stock_series) < 2:
            continue
        
        # If the symbol is the reference, keep it at (100,100).
        if ref_input != "ONE" and sym == ref_input:
            rs_ratio, raw_mom = compute_jdk_indicators(stock_series, stock_series)
        else:
            rs_ratio, raw_mom = compute_jdk_indicators(stock_series, common_ref)
        if rs_ratio.empty or raw_mom.empty:
            continue
        
        if ref_input == "ONE" or ref_raw_mom is None:
            norm_mom = raw_mom * 100.0
        else:
            ref_mom_aligned = ref_raw_mom.reindex(raw_mom.index).dropna()
            if ref_mom_aligned.empty:
                continue
            raw_mom_aligned = raw_mom.reindex(ref_mom_aligned.index).dropna()
            if raw_mom_aligned.empty:
                continue
            norm_mom = (raw_mom_aligned / ref_mom_aligned) * 100.0

        df_temp = pd.DataFrame({
            'Date': rs_ratio.index,
            'Symbol': sym,
            'RS_Ratio': rs_ratio.values
        })
        norm_mom = norm_mom.reindex(rs_ratio.index).ffill().bfill()
        df_temp['RS_Momentum'] = norm_mom.values
        rrg_rows.append(df_temp)
    
    if not rrg_rows:
        return pd.DataFrame()
    
    rrg_df = pd.concat(rrg_rows, ignore_index=True)
    
    # Force the reference stock's values to (100,100) so it is placed in the middle.
    if ref_input != "ONE" and ref_input in rrg_df['Symbol'].unique():
        rrg_df.loc[rrg_df['Symbol'] == ref_input, 'RS_Ratio'] = 100.0
        rrg_df.loc[rrg_df['Symbol'] == ref_input, 'RS_Momentum'] = 100.0

    # Only add the $ONE anchor if reference == "ONE"
    if ref_input == "ONE":
        all_dates = rrg_df['Date'].unique()
        ref_data = [[d, '$ONE', 100.0, 100.0] for d in all_dates]
        ref_df = pd.DataFrame(ref_data, columns=['Date', 'Symbol', 'RS_Ratio', 'RS_Momentum'])
        rrg_df = pd.concat([rrg_df, ref_df], ignore_index=True)
    
    rrg_df = rrg_df.dropna(subset=['RS_Ratio', 'RS_Momentum'])
    return rrg_df

def create_rrg_plotly(rrg_df, trail_length=5):
    """
    Create an interactive Plotly RRG with animation.
    - The reference symbol is forced to (100,100).
    - We compute axis bounds around (100,100) so that if data is close to 100, we "zoom in";
      if data is far from 100, we "zoom out".
    - Each symbol's marker and trail line are colored based on the quadrant of its latest point:
          * Green: RS_Ratio >= 100 and RS_Momentum >= 100
          * Orange: RS_Ratio >= 100 and RS_Momentum < 100
          * Blue: RS_Ratio < 100 and RS_Momentum >= 100
          * Red: RS_Ratio < 100 and RS_Momentum < 100
    """
    if rrg_df.empty:
        fig = go.Figure()
        fig.update_layout(title="No Data Available")
        return fig

    symbols = rrg_df["Symbol"].unique()
    dates = sorted(rrg_df["Date"].unique())

    # Compute the min/max from the data
    x_min_data = rrg_df['RS_Ratio'].min()
    x_max_data = rrg_df['RS_Ratio'].max()
    y_min_data = rrg_df['RS_Momentum'].min()
    y_max_data = rrg_df['RS_Momentum'].max()

    # Determine the maximum distance from 100 in each direction.
    dist_left   = abs(100 - x_min_data)
    dist_right  = abs(x_max_data - 100)
    dist_bottom = abs(100 - y_min_data)
    dist_top    = abs(y_max_data - 100)

    delta = max(dist_left, dist_right, dist_bottom, dist_top)
    delta += 0.2  # add small padding
    min_delta = 1.0  # enforce a minimum delta
    if delta < min_delta:
        delta = min_delta

    x_min = 100 - delta
    x_max = 100 + delta
    y_min = 100 - delta
    y_max = 100 + delta

    fig = go.Figure()
    frames = []

    # Build animation frames
    for dt in dates:
        frame_data = []
        for symbol in symbols:
            sym_data = rrg_df[rrg_df["Symbol"] == symbol].sort_values("Date")
            sym_data = sym_data[sym_data["Date"] <= dt]
            sym_data_trail = sym_data.tail(trail_length)
            if not sym_data_trail.empty:
                # Determine color based on the last point's quadrant.
                last_point = sym_data_trail.iloc[-1]
                x_val = last_point["RS_Ratio"]
                y_val = last_point["RS_Momentum"]
                if x_val >= 100:
                    if y_val >= 100:
                        color = "green"
                    else:
                        color = "orange"
                else:
                    if y_val >= 100:
                        color = "blue"
                    else:
                        color = "red"

                # Marker trace (latest point) with text label
                frame_data.append(
                    go.Scatter(
                        x=[x_val],
                        y=[y_val],
                        mode="markers+text",
                        text=[symbol],
                        textposition="top center",
                        marker=dict(size=8, color=color),
                        name=symbol,
                        showlegend=False
                    )
                )
                # Trail line trace with the same color
                frame_data.append(
                    go.Scatter(
                        x=sym_data_trail["RS_Ratio"],
                        y=sym_data_trail["RS_Momentum"],
                        mode="lines",
                        line=dict(width=2, shape="spline", color=color),
                        name=symbol + " trail",
                        showlegend=False
                    )
                )
        frames.append(go.Frame(data=frame_data, name=str(dt)))

    if frames:
        fig.add_traces(frames[0].data)

    # Set symmetrical axis range around (100,100)
    fig.update_layout(
        xaxis=dict(range=[x_min, x_max]),
        yaxis=dict(range=[y_min, y_max])
    )

    # Quadrant shading with center at (100,100)
    fig.add_shape(
        type="rect", xref="x", yref="y",
        x0=100, x1=200,
        y0=100, y1=200,
        fillcolor="green", opacity=0.1, layer="below", line_width=0
    )
    fig.add_shape(
        type="rect", xref="x", yref="y",
        x0=100, x1=200,
        y0=0, y1=100,
        fillcolor="orange", opacity=0.1, layer="below", line_width=0
    )
    fig.add_shape(
        type="rect", xref="x", yref="y",
        x0=0, x1=100,
        y0=0, y1=100,
        fillcolor="red", opacity=0.1, layer="below", line_width=0
    )
    fig.add_shape(
        type="rect", xref="x", yref="y",
        x0=0, x1=100,
        y0=100, y1=200,
        fillcolor="blue", opacity=0.1, layer="below", line_width=0
    )

    # Add Play/Pause buttons
    fig.update_layout(
        title="Relative Rotation Graph (RRG)",
        xaxis_title="JdK RS Ratio",
        yaxis_title="JdK RS Momentum",
        hovermode="closest",
        updatemenus=[{
            "type": "buttons",
            "showactive": False,
            "buttons": [
                {
                    "label": "Play",
                    "method": "animate",
                    "args": [None, {
                        "frame": {"duration": 500, "redraw": True},
                        "fromcurrent": True,
                        "transition": {"duration": 300},
                        "mode": "immediate"
                    }]
                },
                {
                    "label": "Pause",
                    "method": "animate",
                    "args": [[None], {
                        "frame": {"duration": 0, "redraw": False},
                        "mode": "immediate",
                        "transition": {"duration": 0}
                    }]
                }
            ]
        }]
    )

    # Add slider
    sliders = [{
        "steps": [
            {
                "method": "animate",
                "args": [
                    [frame.name],
                    {
                        "mode": "immediate",
                        "frame": {"duration": 300, "redraw": True},
                        "transition": {"duration": 300}
                    }
                ],
                "label": frame.name,
            }
            for frame in frames
        ],
        "transition": {"duration": 0},
        "x": 0,
        "y": 0,
        "xanchor": "left",
        "yanchor": "top",
        "len": 1.0,
    }]
    fig.update_layout(sliders=sliders)
    fig.frames = frames

    # Increase overall figure size and reduce margins
    fig.update_layout(
        width=1400,
        height=900,
        margin=dict(l=20, r=20, t=60, b=20)
    )
    
    return fig

    """
    Create an interactive Plotly RRG with animation.
    - The reference symbol is forced to (100,100).
    - We compute axis bounds around (100,100) so that if data is close to 100, we "zoom in";
      if data is far from 100, we "zoom out".
    - This ensures the data occupies more of the chart area (rather than looking cramped).
    """
    if rrg_df.empty:
        fig = go.Figure()
        fig.update_layout(title="No Data Available")
        return fig

    symbols = rrg_df["Symbol"].unique()
    dates = sorted(rrg_df["Date"].unique())

    # Compute the min/max from the data
    x_min_data = rrg_df['RS_Ratio'].min()
    x_max_data = rrg_df['RS_Ratio'].max()
    y_min_data = rrg_df['RS_Momentum'].min()
    y_max_data = rrg_df['RS_Momentum'].max()

    # Distance from 100 in each direction
    dist_left   = abs(100 - x_min_data)
    dist_right  = abs(x_max_data - 100)
    dist_bottom = abs(100 - y_min_data)
    dist_top    = abs(y_max_data - 100)

    # We'll use the largest distance among x or y
    # If the data is close to 100, this distance can be small
    # so we add a tiny padding (0.2). Also set a minimum (e.g. 1.0).
    delta = max(dist_left, dist_right, dist_bottom, dist_top)
    delta += 0.2  # small padding
    min_delta = 1.0  # ensure at least a 2-point range
    if delta < min_delta:
        delta = min_delta

    x_min = 100 - delta
    x_max = 100 + delta
    y_min = 100 - delta
    y_max = 100 + delta

    fig = go.Figure()
    frames = []

    # Build animation frames
    for dt in dates:
        frame_data = []
        for symbol in symbols:
            sym_data = rrg_df[rrg_df["Symbol"] == symbol].sort_values("Date")
            sym_data = sym_data[sym_data["Date"] <= dt]
            sym_data_trail = sym_data.tail(trail_length)
            if not sym_data_trail.empty:
                # Marker at the latest point
                frame_data.append(
                    go.Scatter(
                        x=[sym_data_trail.iloc[-1]["RS_Ratio"]],
                        y=[sym_data_trail.iloc[-1]["RS_Momentum"]],
                        mode="markers+text",
                        text=[symbol],
                        textposition="top center",
                        marker=dict(size=8),
                        name=symbol,
                        showlegend=False
                    )
                )
                # Trail line
                frame_data.append(
                    go.Scatter(
                        x=sym_data_trail["RS_Ratio"],
                        y=sym_data_trail["RS_Momentum"],
                        mode="lines",
                        line=dict(width=2, shape="spline"),
                        name=symbol + " trail",
                        showlegend=False
                    )
                )
        frames.append(go.Frame(data=frame_data, name=str(dt)))

    if frames:
        fig.add_traces(frames[0].data)

    # Symmetrical axis range around (100,100)
    fig.update_layout(
        xaxis=dict(range=[x_min, x_max]),
        yaxis=dict(range=[y_min, y_max])
    )

    # Quadrant shading with center at (100,100)
    fig.add_shape(
        type="rect", xref="x", yref="y",
        x0=100, x1=200,
        y0=100, y1=200,
        fillcolor="green", opacity=0.1, layer="below", line_width=0
    )
    fig.add_shape(
        type="rect", xref="x", yref="y",
        x0=100, x1=200,
        y0=0, y1=100,
        fillcolor="orange", opacity=0.1, layer="below", line_width=0
    )
    fig.add_shape(
        type="rect", xref="x", yref="y",
        x0=0, x1=100,
        y0=0, y1=100,
        fillcolor="red", opacity=0.1, layer="below", line_width=0
    )
    fig.add_shape(
        type="rect", xref="x", yref="y",
        x0=0, x1=100,
        y0=100, y1=200,
        fillcolor="blue", opacity=0.1, layer="below", line_width=0
    )

    # Add Play/Pause buttons
    fig.update_layout(
        title="Relative Rotation Graph (RRG)",
        xaxis_title="JdK RS Ratio",
        yaxis_title="JdK RS Momentum",
        hovermode="closest",
        updatemenus=[{
            "type": "buttons",
            "showactive": False,
            "buttons": [
                {
                    "label": "Play",
                    "method": "animate",
                    "args": [None, {
                        "frame": {"duration": 500, "redraw": True},
                        "fromcurrent": True,
                        "transition": {"duration": 300},
                        "mode": "immediate"
                    }]
                },
                {
                    "label": "Pause",
                    "method": "animate",
                    "args": [[None], {
                        "frame": {"duration": 0, "redraw": False},
                        "mode": "immediate",
                        "transition": {"duration": 0}
                    }]
                }
            ]
        }]
    )

    # Add a slider to scrub through time
    sliders = [{
        "steps": [
            {
                "method": "animate",
                "args": [
                    [frame.name],
                    {
                        "mode": "immediate",
                        "frame": {"duration": 300, "redraw": True},
                        "transition": {"duration": 300}
                    }
                ],
                "label": frame.name,
            }
            for frame in frames
        ],
        "transition": {"duration": 0},
        "x": 0,
        "y": 0,
        "xanchor": "left",
        "yanchor": "top",
        "len": 1.0,
    }]
    fig.update_layout(sliders=sliders)
    fig.frames = frames

    # Make the figure bigger and reduce margins
    fig.update_layout(
        width=1200,
        height=600,
        margin=dict(l=20, r=20, t=60, b=20)
    )
    
    return fig

############################
# Streamlit App
############################

def main():
    st.title("Interactive Relative Rotation Graph (RRG)")

    # Choose input method
    input_method = st.radio("Select symbols input method:", options=["Custom Input", "Predefined List"])
    
    if input_method == "Custom Input":
        symbols_input = st.text_input(
            "Enter stock symbols (comma-separated):",
            value="AAPL, MSFT, GOOGL, TSLA, QQQ"
        )
        symbols_list = [s.strip().upper() for s in symbols_input.split(",") if s.strip()]
    else:
        predefined_choice = st.selectbox("Choose a predefined list:", list(predefined_lists.keys()))
        symbols_list = predefined_lists[predefined_choice]
        st.write("Using predefined symbols: " + ", ".join(symbols_list))
    
    reference_input = st.text_input(
        "Enter reference symbol (e.g. SPY, QQQ, or 'ONE'):",
        value="SPY"
    )
    
    data_period_options = ["3 months", "6 months", "1 year", "2 years", "5 years", "10 years", "All time"]
    data_period = st.selectbox("Select data period:", data_period_options)
    
    trail_length_options = [7, 14, 21, 28]
    selected_trail_length = st.selectbox("Select trail length (days):", trail_length_options)
    
    if data_period == "All time":
        start_date = pd.to_datetime("1900-01-01")
    else:
        days_back = period_mapping[data_period]
        start_date = date.today() - timedelta(days=days_back)
    end_date = date.today()
    
    st.write(f"Fetching data from {start_date} to {end_date}...")
    
    generate_button = st.button("Generate RRG")
    if generate_button:
        if not symbols_list:
            st.warning("Please enter at least one symbol.")
            return
        
        to_fetch = symbols_list.copy()
        ref_upper = reference_input.upper()
        if ref_upper != "ONE" and ref_upper not in to_fetch:
            to_fetch.append(ref_upper)
        
        data_dict = fetch_stock_data(to_fetch, start=start_date, end=end_date, interval='1d')
        if not data_dict and ref_upper != "ONE":
            st.warning("No data returned. Check symbols or the selected data period.")
            return
        
        df_prices = combine_data_to_df(data_dict, resample_freq=None)
        df_prices.dropna(axis=0, how='any', inplace=True)
        if df_prices.empty and ref_upper != "ONE":
            st.warning("No data available after combining and dropping missing rows.")
            return
        
        rrg_df = prepare_rrg_dataframe(symbols_list, df_prices, ref_upper)
        if rrg_df.empty:
            st.warning("No RRG data available. Not enough overlapping data for the chosen symbols.")
            return
        
        fig = create_rrg_plotly(rrg_df, trail_length=selected_trail_length)
        st.plotly_chart(fig, use_container_width=False)

if __name__ == "__main__":
    main()
