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

# Custom CSS for dark mode and styling
st.markdown("""
<style>
    .main {
        background-color: #0e1117;
    }
    .stMetric {
        background-color: #1e2130;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #2e3348;
    }
    .stMetric label {
        color: #8b92b0 !important;
    }
    .stMetric .metric-value {
        color: #ffffff !important;
    }
    .css-1d391kg {
        background-color: #1e2130;
    }
    h1, h2, h3 {
        color: #ffffff !important;
    }
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: #1e2130;
        color: #8b92b0;
        text-align: center;
        padding: 10px;
        font-size: 14px;
        border-top: 1px solid #2e3348;
    }
    .contango {
        color: #ef5350 !important;
        font-weight: bold;
    }
    .backwardation {
        color: #26a69a !important;
        font-weight: bold;
    }
    .badge {
        padding: 5px 15px;
        border-radius: 20px;
        font-weight: bold;
        display: inline-block;
    }
    .badge-contango {
        background-color: #ef5350;
        color: white;
    }
    .badge-backwardation {
        background-color: #26a69a;
        color: white;
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
    """Plot VIX term structure curve"""
    fig = go.Figure()
    
    # Main date
    date_data = df[df['data_date'] == selected_date].sort_values('days_to_expiration')
    
    if not date_data.empty:
        state, slope, vix1, vix2 = calculate_market_state(df, selected_date)
        color = '#ef5350' if state == 'Contango' else '#26a69a'
        
        fig.add_trace(go.Scatter(
            x=date_data['days_to_expiration'],
            y=date_data['price'],
            mode='lines+markers',
            name=f"{selected_date.strftime('%Y-%m-%d')} ({state})",
            line=dict(color=color, width=3),
            marker=dict(size=8, color=color),
            hovertemplate='<b>Days to Exp:</b> %{x}<br><b>Price:</b> %{y:.2f}<extra></extra>'
        ))
        
        # Add spot VIX reference (if we have it, otherwise use VIX1)
        if show_spot:
            spot_vix = date_data.iloc[0]['price']
            fig.add_hline(
                y=spot_vix, 
                line_dash="dash", 
                line_color="#ffb74d",
                annotation_text=f"VIX1: {spot_vix:.2f}",
                annotation_position="right"
            )
    
    # Comparison dates
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
                    line=dict(color=colors[i % len(colors)], width=2, dash='dot'),
                    marker=dict(size=6),
                    opacity=0.7,
                    hovertemplate='<b>Days to Exp:</b> %{x}<br><b>Price:</b> %{y:.2f}<extra></extra>'
                ))
    
    fig.update_layout(
        title={
            'text': f"VIX Term Structure - {selected_date.strftime('%B %d, %Y')}",
            'font': {'size': 24, 'color': '#ffffff'}
        },
        xaxis_title="Days to Expiration",
        yaxis_title="VIX Futures Price",
        plot_bgcolor='#1e2130',
        paper_bgcolor='#0e1117',
        font=dict(color='#8b92b0'),
        xaxis=dict(
            gridcolor='#2e3348',
            showgrid=True,
            zeroline=False
        ),
        yaxis=dict(
            gridcolor='#2e3348',
            showgrid=True,
            zeroline=False
        ),
        hovermode='x unified',
        legend=dict(
            bgcolor='#1e2130',
            bordercolor='#2e3348',
            borderwidth=1
        ),
        height=500
    )
    
    return fig

def plot_historical_spread(stats_df, window=20):
    """Plot historical M1-M2 spread with rolling average"""
    fig = go.Figure()
    
    stats_df = stats_df.sort_values('date')
    stats_df['spread_ma'] = stats_df['spread'].rolling(window=window).mean()
    
    # Color based on contango/backwardation
    colors = ['#ef5350' if s == 'Contango' else '#26a69a' for s in stats_df['state']]
    
    fig.add_trace(go.Scatter(
        x=stats_df['date'],
        y=stats_df['spread'],
        mode='markers',
        name='M1-M2 Spread',
        marker=dict(size=3, color=colors, opacity=0.6),
        hovertemplate='<b>Date:</b> %{x}<br><b>Spread:</b> %{y:.2f}<extra></extra>'
    ))
    
    fig.add_trace(go.Scatter(
        x=stats_df['date'],
        y=stats_df['spread_ma'],
        mode='lines',
        name=f'{window}-Day MA',
        line=dict(color='#ffb74d', width=2),
        hovertemplate='<b>Date:</b> %{x}<br><b>MA:</b> %{y:.2f}<extra></extra>'
    ))
    
    fig.add_hline(y=0, line_dash="dash", line_color="#666", opacity=0.5)
    
    fig.update_layout(
        title="M1-M2 Spread Over Time",
        xaxis_title="Date",
        yaxis_title="Spread (VIX2 - VIX1)",
        plot_bgcolor='#1e2130',
        paper_bgcolor='#0e1117',
        font=dict(color='#8b92b0'),
        xaxis=dict(gridcolor='#2e3348', showgrid=True),
        yaxis=dict(gridcolor='#2e3348', showgrid=True),
        hovermode='x unified',
        legend=dict(bgcolor='#1e2130', bordercolor='#2e3348', borderwidth=1),
        height=400
    )
    
    return fig

def plot_market_state_distribution(stats_df):
    """Plot distribution of contango vs backwardation"""
    state_counts = stats_df['state'].value_counts()
    
    fig = go.Figure(data=[
        go.Bar(
            x=state_counts.index,
            y=state_counts.values,
            marker_color=['#ef5350' if x == 'Contango' else '#26a69a' for x in state_counts.index],
            text=state_counts.values,
            textposition='auto',
            hovertemplate='<b>%{x}</b><br>Count: %{y}<br>Percentage: %{customdata:.1f}%<extra></extra>',
            customdata=[(v/state_counts.sum()*100) for v in state_counts.values]
        )
    ])
    
    fig.update_layout(
        title="Market State Distribution",
        xaxis_title="State",
        yaxis_title="Number of Days",
        plot_bgcolor='#1e2130',
        paper_bgcolor='#0e1117',
        font=dict(color='#8b92b0'),
        xaxis=dict(gridcolor='#2e3348'),
        yaxis=dict(gridcolor='#2e3348', showgrid=True),
        height=350
    )
    
    return fig

def plot_spread_distribution(stats_df):
    """Plot distribution of M1-M2 spreads"""
    fig = go.Figure()
    
    fig.add_trace(go.Histogram(
        x=stats_df['spread'],
        nbinsx=50,
        marker_color='#64b5f6',
        opacity=0.7,
        name='Distribution',
        hovertemplate='<b>Spread Range:</b> %{x}<br><b>Count:</b> %{y}<extra></extra>'
    ))
    
    # Add mean line
    mean_spread = stats_df['spread'].mean()
    fig.add_vline(x=mean_spread, line_dash="dash", line_color="#ffb74d", 
                  annotation_text=f"Mean: {mean_spread:.2f}")
    
    fig.update_layout(
        title="M1-M2 Spread Distribution",
        xaxis_title="Spread (VIX2 - VIX1)",
        yaxis_title="Frequency",
        plot_bgcolor='#1e2130',
        paper_bgcolor='#0e1117',
        font=dict(color='#8b92b0'),
        xaxis=dict(gridcolor='#2e3348', showgrid=True),
        yaxis=dict(gridcolor='#2e3348', showgrid=True),
        height=350
    )
    
    return fig

def plot_roll_yield_history(df, lookback_days=252):
    """Plot historical roll yield over time"""
    dates = sorted(df['data_date'].unique())[-lookback_days:]
    
    roll_yields = []
    for date in dates:
        daily, monthly, annual = calculate_roll_yield(df, date)
        if daily is not None:
            roll_yields.append({
                'date': date,
                'daily': daily,
                'monthly': monthly * 100,  # Convert to percentage
                'annual': annual * 100
            })
    
    ry_df = pd.DataFrame(roll_yields)
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=ry_df['date'],
        y=ry_df['monthly'],
        mode='lines',
        name='Monthly Roll Yield %',
        line=dict(color='#ba68c8', width=2),
        fill='tozeroy',
        fillcolor='rgba(186, 104, 200, 0.1)',
        hovertemplate='<b>Date:</b> %{x}<br><b>Monthly:</b> %{y:.2f}%<extra></extra>'
    ))
    
    fig.add_hline(y=0, line_dash="dash", line_color="#666", opacity=0.5)
    
    fig.update_layout(
        title=f"Roll Yield History (Last {lookback_days} Days)",
        xaxis_title="Date",
        yaxis_title="Monthly Roll Yield (%)",
        plot_bgcolor='#1e2130',
        paper_bgcolor='#0e1117',
        font=dict(color='#8b92b0'),
        xaxis=dict(gridcolor='#2e3348', showgrid=True),
        yaxis=dict(gridcolor='#2e3348', showgrid=True),
        hovermode='x unified',
        height=400
    )
    
    return fig

def plot_zscore_analysis(stats_df, window=252):
    """Plot z-score of M1-M2 spread"""
    stats_df = stats_df.sort_values('date').copy()
    stats_df['spread_mean'] = stats_df['spread'].rolling(window=window).mean()
    stats_df['spread_std'] = stats_df['spread'].rolling(window=window).std()
    stats_df['zscore'] = (stats_df['spread'] - stats_df['spread_mean']) / stats_df['spread_std']
    
    fig = go.Figure()
    
    # Color based on z-score magnitude
    colors = ['#ef5350' if z > 2 else '#26a69a' if z < -2 else '#64b5f6' 
              for z in stats_df['zscore'].fillna(0)]
    
    fig.add_trace(go.Scatter(
        x=stats_df['date'],
        y=stats_df['zscore'],
        mode='markers',
        name='Z-Score',
        marker=dict(size=3, color=colors, opacity=0.6),
        hovertemplate='<b>Date:</b> %{x}<br><b>Z-Score:</b> %{y:.2f}<extra></extra>'
    ))
    
    # Add threshold lines
    fig.add_hline(y=2, line_dash="dash", line_color="#ef5350", opacity=0.5, 
                  annotation_text="Extreme Contango (+2σ)")
    fig.add_hline(y=-2, line_dash="dash", line_color="#26a69a", opacity=0.5,
                  annotation_text="Extreme Backwardation (-2σ)")
    fig.add_hline(y=0, line_dash="solid", line_color="#666", opacity=0.3)
    
    fig.update_layout(
        title=f"Term Structure Z-Score ({window}-Day Rolling)",
        xaxis_title="Date",
        yaxis_title="Z-Score",
        plot_bgcolor='#1e2130',
        paper_bgcolor='#0e1117',
        font=dict(color='#8b92b0'),
        xaxis=dict(gridcolor='#2e3348', showgrid=True),
        yaxis=dict(gridcolor='#2e3348', showgrid=True),
        height=400
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
    # Header
    st.markdown("<h1 style='text-align: center; color: #ffffff;'>📊 VIX Term Structure Dashboard</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #8b92b0;'>Advanced Volatility Analysis & Trading Insights</p>", unsafe_allow_html=True)
    st.markdown("---")
    
    # Load data
    with st.spinner('Loading VIX data...'):
        df = load_data()
        stats_df = calculate_historical_stats(df)
    
    # Sidebar
    st.sidebar.image("https://via.placeholder.com/300x80/1e2130/ffffff?text=VIX+Dashboard", use_container_width=True)
    st.sidebar.markdown("---")
    
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
    
    # Top metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        badge_class = "badge-contango" if current_state == "Contango" else "badge-backwardation"
        st.markdown(f"""
        <div style='text-align: center;'>
            <p style='color: #8b92b0; font-size: 14px; margin-bottom: 5px;'>Market State</p>
            <div class='badge {badge_class}'>{current_state.upper()}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.metric(
            "VIX1 (Front Month)",
            f"{vix1:.2f}",
            delta=None
        )
    
    with col3:
        st.metric(
            "VIX2 (Second Month)",
            f"{vix2:.2f}",
            delta=f"{slope:+.2f}" if slope != 0 else "0.00"
        )
    
    with col4:
        if monthly_ry is not None:
            st.metric(
                "Monthly Roll Yield",
                f"{monthly_ry*100:.2f}%",
                delta=None,
                delta_color="inverse"
            )
        else:
            st.metric("Monthly Roll Yield", "N/A")
    
    with col5:
        if annual_ry is not None:
            st.metric(
                "Annual Roll Yield",
                f"{annual_ry*100:.1f}%",
                delta=None,
                delta_color="inverse"
            )
        else:
            st.metric("Annual Roll Yield", "N/A")
    
    st.markdown("---")
    
    # Main chart
    st.subheader("📈 Term Structure Curve")
    fig_ts = plot_term_structure(df, selected_date, comparison_dates if enable_comparison else None, show_spot_vix)
    st.plotly_chart(fig_ts, use_container_width=True)
    
    # Tabs for different analyses
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
