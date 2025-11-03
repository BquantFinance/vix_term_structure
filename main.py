import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="VIX Term Structure Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced CSS for beautiful dark mode with gradients and glass-morphism
st.markdown("""
<style>
    /* Main background with subtle gradient */
    .main {
        background: linear-gradient(135deg, #0a0e1a 0%, #1a1f35 100%);
    }
    
    /* Block container styling */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
    }
    
    /* Enhanced metrics with glass-morphism effect */
    .stMetric {
        background: rgba(30, 33, 48, 0.6);
        backdrop-filter: blur(10px);
        padding: 20px;
        border-radius: 15px;
        border: 1px solid rgba(100, 181, 246, 0.1);
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        transition: all 0.3s ease;
    }
    
    .stMetric:hover {
        transform: translateY(-2px);
        border: 1px solid rgba(100, 181, 246, 0.3);
        box-shadow: 0 12px 40px 0 rgba(0, 0, 0, 0.5);
    }
    
    .stMetric label {
        color: #a0a8c5 !important;
        font-size: 0.85rem !important;
        font-weight: 500 !important;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .stMetric .metric-value {
        color: #ffffff !important;
        font-size: 2rem !important;
        font-weight: 700 !important;
    }
    
    /* Headers with gradient text */
    h1 {
        background: linear-gradient(135deg, #64b5f6 0%, #ba68c8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-weight: 800 !important;
        letter-spacing: -0.5px;
    }
    
    h2, h3 {
        color: #ffffff !important;
        font-weight: 600 !important;
    }
    
    /* Sidebar styling */
    .css-1d391kg, [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1f35 0%, #0a0e1a 100%);
    }
    
    /* Sidebar header */
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
        color: #64b5f6 !important;
    }
    
    /* Enhanced footer with gradient border */
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background: rgba(26, 31, 53, 0.95);
        backdrop-filter: blur(10px);
        color: #8b92b0;
        text-align: center;
        padding: 12px;
        font-size: 14px;
        border-top: 2px solid transparent;
        background-image: 
            linear-gradient(rgba(26, 31, 53, 0.95), rgba(26, 31, 53, 0.95)),
            linear-gradient(90deg, #64b5f6, #ba68c8, #ef5350);
        background-origin: border-box;
        background-clip: padding-box, border-box;
        z-index: 999;
    }
    
    .footer a {
        color: #64b5f6;
        text-decoration: none;
        font-weight: 600;
        transition: color 0.3s ease;
    }
    
    .footer a:hover {
        color: #ba68c8;
    }
    
    /* Badge styling with glow effect */
    .badge {
        padding: 8px 20px;
        border-radius: 25px;
        font-weight: 700;
        display: inline-block;
        font-size: 1rem;
        letter-spacing: 0.5px;
        text-transform: uppercase;
        box-shadow: 0 4px 15px 0 rgba(0, 0, 0, 0.4);
        transition: all 0.3s ease;
    }
    
    .badge:hover {
        transform: scale(1.05);
    }
    
    .badge-contango {
        background: linear-gradient(135deg, #ef5350 0%, #e91e63 100%);
        color: white;
        box-shadow: 0 4px 20px rgba(239, 83, 80, 0.4);
    }
    
    .badge-backwardation {
        background: linear-gradient(135deg, #26a69a 0%, #00897b 100%);
        color: white;
        box-shadow: 0 4px 20px rgba(38, 166, 154, 0.4);
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: rgba(30, 33, 48, 0.4);
        padding: 8px;
        border-radius: 15px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: transparent;
        border-radius: 10px;
        padding: 12px 24px;
        color: #8b92b0;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background-color: rgba(100, 181, 246, 0.1);
        color: #64b5f6;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, rgba(100, 181, 246, 0.2) 0%, rgba(186, 104, 200, 0.2) 100%);
        color: #64b5f6 !important;
    }
    
    /* Button styling */
    .stDownloadButton button {
        background: linear-gradient(135deg, #64b5f6 0%, #ba68c8 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 12px 24px;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(100, 181, 246, 0.3);
    }
    
    .stDownloadButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 25px rgba(100, 181, 246, 0.5);
    }
    
    /* Dataframe styling */
    .dataframe {
        border-radius: 10px;
        overflow: hidden;
    }
    
    /* Divider */
    hr {
        border: none;
        height: 2px;
        background: linear-gradient(90deg, transparent, #64b5f6, transparent);
        margin: 2rem 0;
    }
    
    /* Info/Success/Error boxes */
    .stAlert {
        background: rgba(30, 33, 48, 0.6);
        backdrop-filter: blur(10px);
        border-radius: 10px;
        border-left: 4px solid #64b5f6;
    }
    
    /* Checkbox styling */
    .stCheckbox {
        color: #a0a8c5;
    }
    
    /* Custom metric card styling */
    .metric-card {
        background: rgba(30, 33, 48, 0.6);
        backdrop-filter: blur(10px);
        padding: 20px;
        border-radius: 15px;
        border: 1px solid rgba(100, 181, 246, 0.1);
        text-align: center;
        transition: all 0.3s ease;
    }
    
    .metric-card:hover {
        border-color: rgba(100, 181, 246, 0.3);
        transform: translateY(-2px);
    }
    
    .metric-title {
        color: #a0a8c5;
        font-size: 0.85rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 8px;
    }
    
    .metric-value {
        color: #ffffff;
        font-size: 2rem;
        font-weight: 700;
    }
    
    /* Sidebar divider */
    [data-testid="stSidebar"] hr {
        background: linear-gradient(90deg, transparent, #64b5f6, transparent);
        height: 1px;
        border: none;
        margin: 1.5rem 0;
    }
    
    /* Loading animation enhancement */
    .stSpinner > div {
        border-top-color: #64b5f6 !important;
    }
    
    /* Selectbox and input styling */
    .stSelectbox, .stDateInput, .stNumberInput {
        background: rgba(30, 33, 48, 0.6);
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load and prepare VIX term structure data"""
    df = pd.read_csv('vix_term_structure_data.csv')
    df['data_date'] = pd.to_datetime(df['data_date'])
    df['expiration_date'] = pd.to_datetime(df['expiration_date'])
    return df

@st.cache_data
def calculate_market_state(df, date):
    """Calculate if market is in contango or backwardation"""
    date_data = df[df['data_date'] == date].sort_values('days_to_expiration')
    if len(date_data) >= 2:
        vix1_price = date_data.iloc[0]['price']
        vix2_price = date_data.iloc[1]['price']
        slope = vix2_price - vix1_price
        return 'Contango' if slope > 0 else 'Backwardation', slope, vix1_price, vix2_price
    return 'Unknown', 0, 0, 0

@st.cache_data
def calculate_historical_stats(df):
    """Calculate historical statistics for term structure"""
    results = []
    for date in df['data_date'].unique():
        state, slope, vix1, vix2 = calculate_market_state(df, date)
        results.append({
            'date': date,
            'state': state,
            'slope': slope,
            'vix1': vix1,
            'vix2': vix2,
            'spread': vix2 - vix1
        })
    return pd.DataFrame(results)

@st.cache_data
def calculate_roll_yield(df, date):
    """Calculate expected roll yield for VIX ETFs"""
    date_data = df[df['data_date'] == date].sort_values('days_to_expiration')
    if len(date_data) >= 2:
        daily_carry = date_data.iloc[0]['daily_carry']
        if not pd.isna(daily_carry):
            monthly_carry = daily_carry * 21  # ~21 trading days per month
            annual_carry = daily_carry * 252  # ~252 trading days per year
            return daily_carry, monthly_carry, annual_carry
    return None, None, None

def plot_term_structure(df, selected_date, comparison_dates=None, show_spot=True):
    """Plot VIX term structure curve with enhanced styling"""
    fig = go.Figure()
    
    # Main date
    date_data = df[df['data_date'] == selected_date].sort_values('days_to_expiration')
    
    if not date_data.empty:
        state, slope, vix1, vix2 = calculate_market_state(df, selected_date)
        color = '#ef5350' if state == 'Contango' else '#26a69a'
        
        # Add filled area under the curve
        fig.add_trace(go.Scatter(
            x=date_data['days_to_expiration'],
            y=date_data['price'],
            fill='tozeroy',
            fillcolor=f'rgba({int(color[1:3], 16)}, {int(color[3:5], 16)}, {int(color[5:7], 16)}, 0.1)',
            line=dict(color=color, width=0),
            showlegend=False,
            hoverinfo='skip'
        ))
        
        # Main line with markers
        fig.add_trace(go.Scatter(
            x=date_data['days_to_expiration'],
            y=date_data['price'],
            mode='lines+markers',
            name=f"{selected_date.strftime('%Y-%m-%d')} ({state})",
            line=dict(color=color, width=4, shape='spline'),
            marker=dict(size=10, color=color, line=dict(color='white', width=2)),
            hovertemplate='<b>Days to Exp:</b> %{x}<br><b>Price:</b> %{y:.2f}<extra></extra>'
        ))
        
        # Add spot VIX reference
        if show_spot:
            spot_vix = date_data.iloc[0]['price']
            fig.add_hline(
                y=spot_vix, 
                line_dash="dash", 
                line_color="#ffb74d",
                line_width=2,
                annotation_text=f"  VIX1: {spot_vix:.2f}  ",
                annotation_position="right",
                annotation=dict(
                    bgcolor="rgba(255, 183, 77, 0.2)",
                    bordercolor="#ffb74d",
                    borderwidth=1,
                    borderpad=4,
                    font=dict(color="#ffb74d", size=12, family="Arial")
                )
            )
    
    # Comparison dates with enhanced styling
    if comparison_dates:
        colors = ['#64b5f6', '#ba68c8', '#4db6ac', '#ff8a65']
        for i, comp_date in enumerate(comparison_dates):
            comp_data = df[df['data_date'] == comp_date].sort_values('days_to_expiration')
            if not comp_data.empty:
                fig.add_trace(go.Scatter(
                    x=comp_data['days_to_expiration'],
                    y=comp_data['price'],
                    mode='lines+markers',
                    name=comp_date.strftime('%Y-%m-%d'),
                    line=dict(color=colors[i % len(colors)], width=3, dash='dot', shape='spline'),
                    marker=dict(size=7, line=dict(color='white', width=1)),
                    opacity=0.8,
                    hovertemplate='<b>Days to Exp:</b> %{x}<br><b>Price:</b> %{y:.2f}<extra></extra>'
                ))
    
    fig.update_layout(
        title={
            'text': f"VIX Term Structure - {selected_date.strftime('%B %d, %Y')}",
            'font': {'size': 22, 'color': '#ffffff', 'family': 'Arial'}
        },
        xaxis_title="Days to Expiration",
        yaxis_title="VIX Futures Price",
        plot_bgcolor='rgba(30, 33, 48, 0.4)',
        paper_bgcolor='rgba(0, 0, 0, 0)',
        font=dict(color='#a0a8c5', family='Arial'),
        xaxis=dict(
            gridcolor='rgba(139, 146, 176, 0.1)',
            showgrid=True,
            zeroline=False,
            tickfont=dict(size=11)
        ),
        yaxis=dict(
            gridcolor='rgba(139, 146, 176, 0.1)',
            showgrid=True,
            zeroline=False,
            tickfont=dict(size=11)
        ),
        hovermode='x unified',
        legend=dict(
            bgcolor='rgba(30, 33, 48, 0.8)',
            bordercolor='rgba(100, 181, 246, 0.3)',
            borderwidth=1,
            font=dict(size=11)
        ),
        height=550,
        margin=dict(t=60, b=60, l=60, r=60)
    )
    
    return fig

def plot_historical_spread(stats_df, window=20):
    """Plot historical M1-M2 spread with rolling average - enhanced styling"""
    fig = go.Figure()
    
    stats_df = stats_df.sort_values('date')
    stats_df['spread_ma'] = stats_df['spread'].rolling(window=window).mean()
    
    # Color based on contango/backwardation
    colors = ['#ef5350' if s == 'Contango' else '#26a69a' for s in stats_df['state']]
    
    # Add scatter with gradient effect
    fig.add_trace(go.Scatter(
        x=stats_df['date'],
        y=stats_df['spread'],
        mode='markers',
        name='M1-M2 Spread',
        marker=dict(
            size=4,
            color=colors,
            opacity=0.5,
            line=dict(width=0)
        ),
        hovertemplate='<b>Date:</b> %{x}<br><b>Spread:</b> %{y:.2f}<extra></extra>'
    ))
    
    # Add gradient-styled moving average
    fig.add_trace(go.Scatter(
        x=stats_df['date'],
        y=stats_df['spread_ma'],
        mode='lines',
        name=f'{window}-Day MA',
        line=dict(
            color='#ffb74d',
            width=3,
            shape='spline'
        ),
        hovertemplate='<b>Date:</b> %{x}<br><b>MA:</b> %{y:.2f}<extra></extra>'
    ))
    
    fig.add_hline(y=0, line_dash="solid", line_color="rgba(139, 146, 176, 0.3)", line_width=1)
    
    fig.update_layout(
        title="M1-M2 Spread Over Time",
        xaxis_title="Date",
        yaxis_title="Spread (VIX2 - VIX1)",
        plot_bgcolor='rgba(30, 33, 48, 0.4)',
        paper_bgcolor='rgba(0, 0, 0, 0)',
        font=dict(color='#a0a8c5', family='Arial'),
        xaxis=dict(gridcolor='rgba(139, 146, 176, 0.1)', showgrid=True),
        yaxis=dict(gridcolor='rgba(139, 146, 176, 0.1)', showgrid=True),
        hovermode='x unified',
        legend=dict(
            bgcolor='rgba(30, 33, 48, 0.8)',
            bordercolor='rgba(100, 181, 246, 0.3)',
            borderwidth=1
        ),
        height=450
    )
    
    return fig

def plot_market_state_distribution(stats_df):
    """Plot distribution of contango vs backwardation - enhanced styling"""
    state_counts = stats_df['state'].value_counts()
    
    colors = ['#ef5350' if x == 'Contango' else '#26a69a' for x in state_counts.index]
    
    fig = go.Figure(data=[
        go.Bar(
            x=state_counts.index,
            y=state_counts.values,
            marker=dict(
                color=colors,
                line=dict(color='white', width=2),
                opacity=0.9
            ),
            text=[f"{v}<br>({v/state_counts.sum()*100:.1f}%)" for v in state_counts.values],
            textposition='auto',
            textfont=dict(size=14, color='white', family='Arial'),
            hovertemplate='<b>%{x}</b><br>Count: %{y}<br>Percentage: %{customdata:.1f}%<extra></extra>',
            customdata=[(v/state_counts.sum()*100) for v in state_counts.values]
        )
    ])
    
    fig.update_layout(
        title="Market State Distribution",
        xaxis_title="State",
        yaxis_title="Number of Days",
        plot_bgcolor='rgba(30, 33, 48, 0.4)',
        paper_bgcolor='rgba(0, 0, 0, 0)',
        font=dict(color='#a0a8c5', family='Arial'),
        xaxis=dict(gridcolor='rgba(139, 146, 176, 0.1)'),
        yaxis=dict(gridcolor='rgba(139, 146, 176, 0.1)', showgrid=True),
        height=380
    )
    
    return fig

def plot_spread_distribution(stats_df):
    """Plot distribution of M1-M2 spreads - enhanced styling"""
    fig = go.Figure()
    
    fig.add_trace(go.Histogram(
        x=stats_df['spread'],
        nbinsx=50,
        marker=dict(
            color='#64b5f6',
            opacity=0.8,
            line=dict(color='white', width=1)
        ),
        name='Distribution',
        hovertemplate='<b>Spread Range:</b> %{x}<br><b>Count:</b> %{y}<extra></extra>'
    ))
    
    # Add mean line with enhanced annotation
    mean_spread = stats_df['spread'].mean()
    fig.add_vline(
        x=mean_spread, 
        line_dash="dash", 
        line_color="#ffb74d",
        line_width=2,
        annotation_text=f"  Mean: {mean_spread:.2f}  ",
        annotation=dict(
            bgcolor="rgba(255, 183, 77, 0.2)",
            bordercolor="#ffb74d",
            borderwidth=1,
            font=dict(color="#ffb74d", size=11)
        )
    )
    
    fig.update_layout(
        title="M1-M2 Spread Distribution",
        xaxis_title="Spread (VIX2 - VIX1)",
        yaxis_title="Frequency",
        plot_bgcolor='rgba(30, 33, 48, 0.4)',
        paper_bgcolor='rgba(0, 0, 0, 0)',
        font=dict(color='#a0a8c5', family='Arial'),
        xaxis=dict(gridcolor='rgba(139, 146, 176, 0.1)', showgrid=True),
        yaxis=dict(gridcolor='rgba(139, 146, 176, 0.1)', showgrid=True),
        height=380
    )
    
    return fig

def plot_roll_yield_history(df, lookback_days=252):
    """Plot historical roll yield over time - enhanced styling"""
    dates = sorted(df['data_date'].unique())[-lookback_days:]
    
    roll_yields = []
    for date in dates:
        daily, monthly, annual = calculate_roll_yield(df, date)
        if daily is not None:
            roll_yields.append({
                'date': date,
                'daily': daily,
                'monthly': monthly * 100,
                'annual': annual * 100
            })
    
    ry_df = pd.DataFrame(roll_yields)
    
    fig = go.Figure()
    
    # Create gradient fill colors based on positive/negative
    fill_colors = ['rgba(239, 83, 80, 0.15)' if y > 0 else 'rgba(38, 166, 154, 0.15)' 
                   for y in ry_df['monthly']]
    
    fig.add_trace(go.Scatter(
        x=ry_df['date'],
        y=ry_df['monthly'],
        mode='lines',
        name='Monthly Roll Yield %',
        line=dict(color='#ba68c8', width=3, shape='spline'),
        fill='tozeroy',
        fillcolor='rgba(186, 104, 200, 0.15)',
        hovertemplate='<b>Date:</b> %{x}<br><b>Monthly:</b> %{y:.2f}%<extra></extra>'
    ))
    
    fig.add_hline(y=0, line_dash="solid", line_color="rgba(139, 146, 176, 0.3)", line_width=2)
    
    fig.update_layout(
        title=f"Roll Yield History (Last {lookback_days} Days)",
        xaxis_title="Date",
        yaxis_title="Monthly Roll Yield (%)",
        plot_bgcolor='rgba(30, 33, 48, 0.4)',
        paper_bgcolor='rgba(0, 0, 0, 0)',
        font=dict(color='#a0a8c5', family='Arial'),
        xaxis=dict(gridcolor='rgba(139, 146, 176, 0.1)', showgrid=True),
        yaxis=dict(gridcolor='rgba(139, 146, 176, 0.1)', showgrid=True),
        hovermode='x unified',
        height=450
    )
    
    return fig

def plot_zscore_analysis(stats_df, window=252):
    """Plot z-score of M1-M2 spread - enhanced styling"""
    stats_df = stats_df.sort_values('date').copy()
    stats_df['spread_mean'] = stats_df['spread'].rolling(window=window).mean()
    stats_df['spread_std'] = stats_df['spread'].rolling(window=window).std()
    stats_df['zscore'] = (stats_df['spread'] - stats_df['spread_mean']) / stats_df['spread_std']
    
    fig = go.Figure()
    
    # Color based on z-score magnitude with gradient
    colors = ['#ef5350' if z > 2 else '#26a69a' if z < -2 else '#64b5f6' 
              for z in stats_df['zscore'].fillna(0)]
    
    fig.add_trace(go.Scatter(
        x=stats_df['date'],
        y=stats_df['zscore'],
        mode='markers',
        name='Z-Score',
        marker=dict(
            size=4,
            color=colors,
            opacity=0.5,
            line=dict(width=0)
        ),
        hovertemplate='<b>Date:</b> %{x}<br><b>Z-Score:</b> %{y:.2f}<extra></extra>'
    ))
    
    # Add threshold lines with enhanced styling
    fig.add_hline(
        y=2, 
        line_dash="dash", 
        line_color="#ef5350", 
        line_width=2,
        opacity=0.6,
        annotation_text="  Extreme Contango (+2σ)  ",
        annotation=dict(
            bgcolor="rgba(239, 83, 80, 0.1)",
            bordercolor="#ef5350",
            borderwidth=1,
            font=dict(color="#ef5350", size=10)
        )
    )
    fig.add_hline(
        y=-2, 
        line_dash="dash", 
        line_color="#26a69a", 
        line_width=2,
        opacity=0.6,
        annotation_text="  Extreme Backwardation (-2σ)  ",
        annotation=dict(
            bgcolor="rgba(38, 166, 154, 0.1)",
            bordercolor="#26a69a",
            borderwidth=1,
            font=dict(color="#26a69a", size=10)
        )
    )
    fig.add_hline(y=0, line_dash="solid", line_color="rgba(139, 146, 176, 0.3)", line_width=1)
    
    fig.update_layout(
        title=f"Term Structure Z-Score ({window}-Day Rolling)",
        xaxis_title="Date",
        yaxis_title="Z-Score",
        plot_bgcolor='rgba(30, 33, 48, 0.4)',
        paper_bgcolor='rgba(0, 0, 0, 0)',
        font=dict(color='#a0a8c5', family='Arial'),
        xaxis=dict(gridcolor='rgba(139, 146, 176, 0.1)', showgrid=True),
        yaxis=dict(gridcolor='rgba(139, 146, 176, 0.1)', showgrid=True),
        height=450
    )
    
    return fig

def plot_heatmap_calendar(df, year):
    """Create a calendar heatmap of market states"""
    year_data = df[df['date'].dt.year == year].copy()
    year_data['month'] = year_data['date'].dt.month
    year_data['day'] = year_data['date'].dt.day
    year_data['state_numeric'] = year_data['state'].map({'Contango': 1, 'Backwardation': -1, 'Unknown': 0})
    
    # Create pivot table
    pivot = year_data.pivot_table(values='state_numeric', index='day', columns='month', aggfunc='mean')
    
    fig = go.Figure(data=go.Heatmap(
        z=pivot.values,
        x=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],
        y=pivot.index,
        colorscale=[[0, '#26a69a'], [0.5, '#666666'], [1, '#ef5350']],
        zmid=0,
        colorbar=dict(
            title="State",
            tickvals=[-1, 0, 1],
            ticktext=['Back', 'Neutral', 'Contango']
        ),
        hovertemplate='<b>Month:</b> %{x}<br><b>Day:</b> %{y}<br><b>State:</b> %{z:.2f}<extra></extra>'
    ))
    
    fig.update_layout(
        title=f"Market State Calendar - {year}",
        xaxis_title="Month",
        yaxis_title="Day",
        plot_bgcolor='#1e2130',
        paper_bgcolor='#0e1117',
        font=dict(color='#8b92b0'),
        height=500
    )
    
    return fig

# Main app
def main():
    # Enhanced Header with gradient and icons
    st.markdown("""
    <div style='text-align: center; padding: 2rem 0 1rem 0;'>
        <h1 style='font-size: 3rem; margin-bottom: 0.5rem; background: linear-gradient(135deg, #64b5f6 0%, #ba68c8 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;'>
            📊 VIX Term Structure Dashboard
        </h1>
        <p style='font-size: 1.1rem; color: #a0a8c5; font-weight: 500; letter-spacing: 1px;'>
            Advanced Volatility Analysis & Trading Insights
        </p>
        <div style='width: 100px; height: 3px; background: linear-gradient(90deg, #64b5f6, #ba68c8); margin: 1rem auto; border-radius: 2px;'></div>
    </div>
    """, unsafe_allow_html=True)
    
    # Load data
    with st.spinner('🔄 Loading VIX data...'):
        df = load_data()
        stats_df = calculate_historical_stats(df)
    
    # Enhanced Sidebar Header
    st.sidebar.markdown("""
    <div style='text-align: center; padding: 1.5rem 0; background: rgba(100, 181, 246, 0.1); border-radius: 15px; margin-bottom: 1rem;'>
        <h2 style='margin: 0; background: linear-gradient(135deg, #64b5f6 0%, #ba68c8 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; font-size: 1.5rem;'>
            ⚙️ Controls
        </h2>
    </div>
    """, unsafe_allow_html=True)
    
    # Date selection
    st.sidebar.subheader("📅 Date Selection")
    available_dates = sorted(df['data_date'].unique())
    
    max_date = available_dates[-1]
    selected_date = st.sidebar.date_input(
        "Select Date",
        value=max_date,
        min_value=available_dates[0].date(),
        max_value=max_date.date()
    )
    selected_date = pd.to_datetime(selected_date)
    
    # Find nearest available date
    if selected_date not in available_dates:
        nearest_date = min(available_dates, key=lambda x: abs(x - selected_date))
        st.sidebar.warning(f"Date adjusted to nearest available: {nearest_date.strftime('%Y-%m-%d')}")
        selected_date = nearest_date
    
    # Comparison dates
    st.sidebar.markdown("---")
    st.sidebar.subheader("📊 Comparison Dates")
    enable_comparison = st.sidebar.checkbox("Enable Comparison", value=False)
    
    comparison_dates = []
    if enable_comparison:
        num_comparisons = st.sidebar.slider("Number of comparisons", 1, 4, 2)
        for i in range(num_comparisons):
            comp_date = st.sidebar.date_input(
                f"Comparison {i+1}",
                value=(max_date - timedelta(days=30*(i+1))).date(),
                key=f"comp_{i}"
            )
            comparison_dates.append(pd.to_datetime(comp_date))
    
    # Analysis options
    st.sidebar.markdown("---")
    st.sidebar.subheader("⚙️ Analysis Options")
    show_spot_vix = st.sidebar.checkbox("Show VIX1 Reference Line", value=True)
    rolling_window = st.sidebar.slider("Rolling Window (days)", 10, 100, 20)
    zscore_window = st.sidebar.slider("Z-Score Window (days)", 60, 500, 252)
    
    # Main content
    current_state, slope, vix1, vix2 = calculate_market_state(df, selected_date)
    daily_ry, monthly_ry, annual_ry = calculate_roll_yield(df, selected_date)
    
    # Enhanced top metrics with gradient cards
    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        badge_class = "badge-contango" if current_state == "Contango" else "badge-backwardation"
        badge_emoji = "🔴" if current_state == "Contango" else "🟢"
        st.markdown(f"""
        <div style='text-align: center; padding: 10px;'>
            <p style='color: #a0a8c5; font-size: 0.85rem; margin-bottom: 8px; font-weight: 500; text-transform: uppercase; letter-spacing: 0.5px;'>Market State</p>
            <div class='badge {badge_class}' style='font-size: 0.95rem;'>{badge_emoji} {current_state.upper()}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-title'>📈 VIX1 (Front Month)</div>
            <div class='metric-value'>{vix1:.2f}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        delta_color = "#ef5350" if slope > 0 else "#26a69a" if slope < 0 else "#8b92b0"
        delta_symbol = "▲" if slope > 0 else "▼" if slope < 0 else "●"
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-title'>📊 VIX2 (Second Month)</div>
            <div class='metric-value'>{vix2:.2f}</div>
            <div style='color: {delta_color}; font-size: 0.9rem; margin-top: 4px; font-weight: 600;'>
                {delta_symbol} {slope:+.2f}
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        if monthly_ry is not None:
            ry_color = "#ef5350" if monthly_ry > 0 else "#26a69a"
            st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-title'>📅 Monthly Roll Yield</div>
                <div class='metric-value' style='color: {ry_color};'>{monthly_ry*100:.2f}%</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class='metric-card'>
                <div class='metric-title'>📅 Monthly Roll Yield</div>
                <div class='metric-value' style='color: #8b92b0;'>N/A</div>
            </div>
            """, unsafe_allow_html=True)
    
    with col5:
        if annual_ry is not None:
            ry_color = "#ef5350" if annual_ry > 0 else "#26a69a"
            st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-title'>📆 Annual Roll Yield</div>
                <div class='metric-value' style='color: {ry_color};'>{annual_ry*100:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class='metric-card'>
                <div class='metric-title'>📆 Annual Roll Yield</div>
                <div class='metric-value' style='color: #8b92b0;'>N/A</div>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Enhanced section header
    st.markdown("""
    <div style='margin-bottom: 1.5rem;'>
        <h2 style='display: inline-block; margin: 0; font-size: 1.8rem;'>📈 Term Structure Curve</h2>
        <p style='color: #a0a8c5; margin-top: 0.5rem; font-size: 0.95rem;'>Visualize VIX futures prices across different maturities</p>
    </div>
    """, unsafe_allow_html=True)
    
    fig_ts = plot_term_structure(df, selected_date, comparison_dates if enable_comparison else None, show_spot_vix)
    st.plotly_chart(fig_ts, use_container_width=True)
    
    # Enhanced tabs section
    st.markdown("<br>", unsafe_allow_html=True)
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Historical Spread",
        "🎯 Distribution Analysis",
        "📉 Roll Yield",
        "🔬 Z-Score Analysis",
        "📅 Calendar View"
    ])
    
    with tab1:
        st.subheader("M1-M2 Spread Over Time")
        fig_spread = plot_historical_spread(stats_df, window=rolling_window)
        st.plotly_chart(fig_spread, use_container_width=True)
        
        # Statistics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Mean Spread", f"{stats_df['spread'].mean():.2f}")
        with col2:
            st.metric("Std Dev", f"{stats_df['spread'].std():.2f}")
        with col3:
            contango_pct = (stats_df['state'] == 'Contango').sum() / len(stats_df) * 100
            st.metric("% in Contango", f"{contango_pct:.1f}%")
        with col4:
            backwardation_pct = (stats_df['state'] == 'Backwardation').sum() / len(stats_df) * 100
            st.metric("% in Backwardation", f"{backwardation_pct:.1f}%")
    
    with tab2:
        col1, col2 = st.columns(2)
        with col1:
            fig_state_dist = plot_market_state_distribution(stats_df)
            st.plotly_chart(fig_state_dist, use_container_width=True)
        with col2:
            fig_spread_dist = plot_spread_distribution(stats_df)
            st.plotly_chart(fig_spread_dist, use_container_width=True)
        
        # Percentile analysis
        st.markdown("### 📊 Percentile Analysis")
        current_spread = vix2 - vix1
        percentile = stats.percentileofscore(stats_df['spread'].dropna(), current_spread)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Current Spread", f"{current_spread:.2f}")
        with col2:
            st.metric("Percentile Rank", f"{percentile:.1f}%")
        with col3:
            if percentile > 90:
                st.error("🔴 Extremely High Contango")
            elif percentile < 10:
                st.success("🟢 Extreme Backwardation")
            else:
                st.info("🔵 Normal Range")
    
    with tab3:
        fig_roll = plot_roll_yield_history(df, lookback_days=min(252, len(stats_df)))
        st.plotly_chart(fig_roll, use_container_width=True)
        
        st.markdown("""
        ### 💡 Roll Yield Explanation
        
        **Positive Roll Yield (Contango)**: VIX futures are trading at higher prices than spot VIX. 
        This creates a **headwind** for long VIX ETFs (VXX, UVXY), as they lose value when rolling contracts.
        
        **Negative Roll Yield (Backwardation)**: VIX futures are trading at lower prices than spot VIX.
        This creates a **tailwind** for long VIX ETFs, as they gain value when rolling contracts.
        
        **For VIX ETF Traders**: 
        - High contango (positive roll yield) = Decay accelerates for long positions
        - Backwardation (negative roll yield) = Long positions benefit from roll
        """)
        
        # Roll yield summary table
        if daily_ry is not None:
            st.markdown("### 📋 Current Roll Yield Summary")
            roll_summary = pd.DataFrame({
                'Period': ['Daily', 'Monthly', 'Annual'],
                'Roll Yield (%)': [daily_ry*100, monthly_ry*100, annual_ry*100],
                'Impact on VXX': [
                    'Minimal' if abs(daily_ry) < 0.01 else 'Moderate' if abs(daily_ry) < 0.03 else 'Severe',
                    'Minimal' if abs(monthly_ry) < 0.05 else 'Moderate' if abs(monthly_ry) < 0.15 else 'Severe',
                    'Minimal' if abs(annual_ry) < 0.50 else 'Moderate' if abs(annual_ry) < 1.5 else 'Severe'
                ]
            })
            st.dataframe(roll_summary, use_container_width=True)
    
    with tab4:
        fig_zscore = plot_zscore_analysis(stats_df, window=zscore_window)
        st.plotly_chart(fig_zscore, use_container_width=True)
        
        st.markdown("""
        ### 🎯 Z-Score Interpretation
        
        The z-score measures how unusual the current term structure is compared to historical norms:
        
        - **Z-Score > +2**: Extremely steep contango (rare event, >95th percentile)
        - **Z-Score between -2 and +2**: Normal range
        - **Z-Score < -2**: Extreme backwardation (rare event, <5th percentile)
        
        **Trading Implications**:
        - Extreme readings often precede mean reversion
        - Z-score > +2: Consider short volatility strategies
        - Z-score < -2: Often occurs during market stress/crashes
        """)
        
        # Current z-score
        current_spread = vix2 - vix1
        recent_stats = stats_df.tail(zscore_window)
        if len(recent_stats) >= zscore_window:
            mean_spread = recent_stats['spread'].mean()
            std_spread = recent_stats['spread'].std()
            current_zscore = (current_spread - mean_spread) / std_spread
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Current Z-Score", f"{current_zscore:.2f}")
            with col2:
                st.metric(f"{zscore_window}-Day Mean", f"{mean_spread:.2f}")
            with col3:
                st.metric(f"{zscore_window}-Day Std Dev", f"{std_spread:.2f}")
    
    with tab5:
        st.subheader("Calendar Heatmap")
        selected_year = st.selectbox(
            "Select Year",
            sorted(stats_df['date'].dt.year.unique(), reverse=True)
        )
        
        fig_calendar = plot_heatmap_calendar(stats_df, selected_year)
        st.plotly_chart(fig_calendar, use_container_width=True)
        
        # Year summary
        year_stats = stats_df[stats_df['date'].dt.year == selected_year]
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Trading Days", len(year_stats))
        with col2:
            contango_days = (year_stats['state'] == 'Contango').sum()
            st.metric("Contango Days", contango_days)
        with col3:
            back_days = (year_stats['state'] == 'Backwardation').sum()
            st.metric("Backwardation Days", back_days)
        with col4:
            avg_spread = year_stats['spread'].mean()
            st.metric("Avg Spread", f"{avg_spread:.2f}")
    
    # Data export section
    st.markdown("---")
    st.subheader("📥 Export Data")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Export current term structure
        current_ts = df[df['data_date'] == selected_date].sort_values('days_to_expiration')
        csv_ts = current_ts.to_csv(index=False)
        st.download_button(
            label="Download Current Term Structure",
            data=csv_ts,
            file_name=f"vix_ts_{selected_date.strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )
    
    with col2:
        # Export historical stats
        csv_stats = stats_df.to_csv(index=False)
        st.download_button(
            label="Download Historical Statistics",
            data=csv_stats,
            file_name="vix_historical_stats.csv",
            mime="text/csv"
        )
    
    with col3:
        # Export filtered data
        date_range = st.sidebar.date_input(
            "Export Date Range",
            value=(stats_df['date'].min().date(), stats_df['date'].max().date()),
            key="export_range"
        )
        if len(date_range) == 2:
            filtered_stats = stats_df[
                (stats_df['date'] >= pd.to_datetime(date_range[0])) & 
                (stats_df['date'] <= pd.to_datetime(date_range[1]))
            ]
            csv_filtered = filtered_stats.to_csv(index=False)
            st.download_button(
                label="Download Filtered Range",
                data=csv_filtered,
                file_name=f"vix_data_{date_range[0]}_{date_range[1]}.csv",
                mime="text/csv"
            )
    
    # Footer
    st.markdown("<br><br><br>", unsafe_allow_html=True)
    st.markdown("""
    <div class='footer'>
        Made with ❤️ by <strong>@Gsnchez</strong> | <a href='http://bquantfinance.com' target='_blank' style='color: #64b5f6; text-decoration: none;'>bquantfinance.com</a>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
