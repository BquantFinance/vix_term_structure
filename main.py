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

# Configuración de la página
st.set_page_config(
    page_title="Panel de Estructura de Términos VIX",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS mejorado para modo oscuro con gradientes y glass-morphism
st.markdown("""
<style>
    /* Fondo principal con gradiente sutil */
    .main {
        background: linear-gradient(135deg, #0a0e1a 0%, #1a1f35 100%);
    }
    
    /* Estilo del contenedor de bloques */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
    }
    
    /* Métricas mejoradas con efecto glass-morphism */
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
    
    /* Encabezados con texto en gradiente */
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
    
    /* Estilo de la barra lateral */
    .css-1d391kg, [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1f35 0%, #0a0e1a 100%);
    }
    
    /* Encabezado de la barra lateral */
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
        color: #64b5f6 !important;
    }
    
    /* Pie de página mejorado con borde en gradiente */
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
    
    /* Estilo de insignias con efecto de brillo */
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
    
    /* Estilo de pestañas */
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
    
    /* Estilo de botones */
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
    
    /* Estilo de dataframe */
    .dataframe {
        border-radius: 10px;
        overflow: hidden;
    }
    
    /* Divisor */
    hr {
        border: none;
        height: 2px;
        background: linear-gradient(90deg, transparent, #64b5f6, transparent);
        margin: 2rem 0;
    }
    
    /* Cajas de Info/Success/Error */
    .stAlert {
        background: rgba(30, 33, 48, 0.6);
        backdrop-filter: blur(10px);
        border-radius: 10px;
        border-left: 4px solid #64b5f6;
    }
    
    /* Tarjeta de métrica personalizada */
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
    
    /* Divisor de barra lateral */
    [data-testid="stSidebar"] hr {
        background: linear-gradient(90deg, transparent, #64b5f6, transparent);
        height: 1px;
        border: none;
        margin: 1.5rem 0;
    }
    
    /* Mejora de animación de carga */
    .stSpinner > div {
        border-top-color: #64b5f6 !important;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Cargar y preparar datos de estructura de términos VIX"""
    df = pd.read_csv('vix_term_structure_data.csv')
    df['data_date'] = pd.to_datetime(df['data_date'])
    df['expiration_date'] = pd.to_datetime(df['expiration_date'])
    return df

@st.cache_data
def calculate_market_state(df, date):
    """Calcular si el mercado está en contango o backwardation"""
    date_data = df[df['data_date'] == date].sort_values('days_to_expiration')
    if len(date_data) >= 2:
        vix1_price = date_data.iloc[0]['price']
        vix2_price = date_data.iloc[1]['price']
        slope = vix2_price - vix1_price
        return 'Contango' if slope > 0 else 'Backwardation', slope, vix1_price, vix2_price
    return 'Desconocido', 0, 0, 0

@st.cache_data
def calculate_historical_stats(df):
    """Calcular estadísticas históricas para la estructura de términos"""
    results = []
    for date in df['data_date'].unique():
        state, slope, vix1, vix2 = calculate_market_state(df, date)
        
        # Calcular curvatura (butterfly)
        date_data = df[df['data_date'] == date].sort_values('days_to_expiration')
        butterfly = np.nan
        if len(date_data) >= 3:
            butterfly = date_data.iloc[0]['price'] - 2*date_data.iloc[1]['price'] + date_data.iloc[2]['price']
        
        results.append({
            'date': date,
            'state': state,
            'slope': slope,
            'vix1': vix1,
            'vix2': vix2,
            'spread': vix2 - vix1,
            'butterfly': butterfly
        })
    return pd.DataFrame(results)

@st.cache_data
def calculate_roll_yield(df, date):
    """Calcular rendimiento esperado del roll para ETFs de VIX"""
    date_data = df[df['data_date'] == date].sort_values('days_to_expiration')
    if len(date_data) >= 2:
        daily_carry = date_data.iloc[0]['daily_carry']
        if not pd.isna(daily_carry):
            monthly_carry = daily_carry * 21
            annual_carry = daily_carry * 252
            return daily_carry, monthly_carry, annual_carry
    return None, None, None

@st.cache_data
def calculate_persistence_stats(stats_df):
    """Calcular estadísticas de persistencia de estados de mercado"""
    stats_df = stats_df.sort_values('date')
    stats_df['state_change'] = stats_df['state'] != stats_df['state'].shift(1)
    stats_df['streak_id'] = stats_df['state_change'].cumsum()
    
    streaks = stats_df.groupby(['streak_id', 'state']).size().reset_index(name='duration')
    streaks = streaks[streaks['state'].isin(['Contango', 'Backwardation'])]
    
    return streaks

@st.cache_data
def find_extreme_days(stats_df, n=10):
    """Encontrar los días más extremos en la historia"""
    # Top contango
    top_contango = stats_df.nlargest(n, 'spread')[['date', 'spread', 'vix1', 'vix2']]
    top_contango['type'] = 'Contango Extremo'
    
    # Top backwardation
    top_backwardation = stats_df.nsmallest(n, 'spread')[['date', 'spread', 'vix1', 'vix2']]
    top_backwardation['type'] = 'Backwardation Extremo'
    
    # Mayores cambios
    stats_df['spread_change'] = stats_df['spread'].diff().abs()
    top_changes = stats_df.nlargest(n, 'spread_change')[['date', 'spread', 'spread_change', 'vix1']]
    top_changes['type'] = 'Mayor Cambio'
    
    return top_contango, top_backwardation, top_changes

@st.cache_data
def simulate_vxx_performance(df, start_date, end_date, initial_capital=10000):
    """Simular rendimiento de VXX usando datos de carry diario"""
    filtered_df = df[(df['data_date'] >= start_date) & (df['data_date'] <= end_date)].copy()
    filtered_df = filtered_df.sort_values('data_date')
    
    # Obtener carry diario para VIX1
    daily_carries = []
    dates = []
    
    for date in filtered_df['data_date'].unique():
        date_data = filtered_df[filtered_df['data_date'] == date].sort_values('days_to_expiration')
        if len(date_data) >= 1 and not pd.isna(date_data.iloc[0]['daily_carry']):
            daily_carries.append(date_data.iloc[0]['daily_carry'])
            dates.append(date)
    
    # Simular valor de la cartera
    portfolio_values = [initial_capital]
    for carry in daily_carries:
        # VXX pierde el carry diario en contango, gana en backwardation
        new_value = portfolio_values[-1] * (1 - carry)
        portfolio_values.append(new_value)
    
    return pd.DataFrame({
        'date': [start_date] + dates,
        'value': portfolio_values,
        'return': [0] + [((portfolio_values[i] / portfolio_values[i-1]) - 1) * 100 
                         for i in range(1, len(portfolio_values))]
    })

@st.cache_data
def calculate_seasonality(stats_df):
    """Calcular patrones estacionales"""
    stats_df = stats_df.copy()
    stats_df['month'] = stats_df['date'].dt.month
    stats_df['month_name'] = stats_df['date'].dt.strftime('%B')
    
    monthly_stats = stats_df.groupby('month').agg({
        'spread': ['mean', 'median', 'std'],
        'vix1': 'mean',
        'state': lambda x: (x == 'Contango').sum() / len(x) * 100
    }).reset_index()
    
    monthly_stats.columns = ['month', 'spread_mean', 'spread_median', 'spread_std', 
                            'vix1_mean', 'contango_pct']
    
    month_names = ['Enero', 'Febrero', 'Marzo', 'Abril', 'Mayo', 'Junio',
                   'Julio', 'Agosto', 'Septiembre', 'Octubre', 'Noviembre', 'Diciembre']
    monthly_stats['month_name'] = [month_names[i-1] for i in monthly_stats['month']]
    
    return monthly_stats

def plot_term_structure(df, selected_date, comparison_dates=None, show_spot=True):
    """Graficar curva de estructura de términos VIX con estilo mejorado"""
    fig = go.Figure()
    
    # Fecha principal
    date_data = df[df['data_date'] == selected_date].sort_values('days_to_expiration')
    
    if not date_data.empty:
        state, slope, vix1, vix2 = calculate_market_state(df, selected_date)
        color = '#ef5350' if state == 'Contango' else '#26a69a'
        
        # Añadir área rellena bajo la curva
        fig.add_trace(go.Scatter(
            x=date_data['days_to_expiration'],
            y=date_data['price'],
            fill='tozeroy',
            fillcolor=f'rgba({int(color[1:3], 16)}, {int(color[3:5], 16)}, {int(color[5:7], 16)}, 0.1)',
            line=dict(color=color, width=0),
            showlegend=False,
            hoverinfo='skip'
        ))
        
        # Línea principal con marcadores
        fig.add_trace(go.Scatter(
            x=date_data['days_to_expiration'],
            y=date_data['price'],
            mode='lines+markers',
            name=f"{selected_date.strftime('%Y-%m-%d')} ({state})",
            line=dict(color=color, width=4, shape='spline'),
            marker=dict(size=10, color=color, line=dict(color='white', width=2)),
            hovertemplate='<b>Días hasta venc.:</b> %{x}<br><b>Precio:</b> %{y:.2f}<extra></extra>'
        ))
        
        # Añadir referencia VIX spot
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
    
    # Fechas de comparación con estilo mejorado
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
                    hovertemplate='<b>Días hasta venc.:</b> %{x}<br><b>Precio:</b> %{y:.2f}<extra></extra>'
                ))
    
    fig.update_layout(
        title={
            'text': f"Estructura de Términos VIX - {selected_date.strftime('%d de %B, %Y')}",
            'font': {'size': 22, 'color': '#ffffff', 'family': 'Arial'}
        },
        xaxis_title="Días hasta Vencimiento",
        yaxis_title="Precio de Futuros VIX",
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

def plot_seasonality(monthly_stats):
    """Graficar patrones estacionales"""
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Spread Promedio por Mes', 'Nivel VIX1 Promedio',
                       'Distribución de Spread', '% Días en Contango'),
        specs=[[{"type": "bar"}, {"type": "bar"}],
               [{"type": "box"}, {"type": "bar"}]]
    )
    
    # Spread promedio
    fig.add_trace(
        go.Bar(
            x=monthly_stats['month_name'],
            y=monthly_stats['spread_mean'],
            marker_color='#64b5f6',
            name='Spread Promedio',
            hovertemplate='<b>%{x}</b><br>Spread: %{y:.2f}<extra></extra>'
        ),
        row=1, col=1
    )
    
    # VIX1 promedio
    fig.add_trace(
        go.Bar(
            x=monthly_stats['month_name'],
            y=monthly_stats['vix1_mean'],
            marker_color='#ba68c8',
            name='VIX1 Promedio',
            hovertemplate='<b>%{x}</b><br>VIX1: %{y:.2f}<extra></extra>'
        ),
        row=1, col=2
    )
    
    # Box plot de spread
    fig.add_trace(
        go.Box(
            x=monthly_stats['month_name'],
            y=monthly_stats['spread_mean'],
            marker_color='#26a69a',
            name='Distribución',
            hovertemplate='<b>%{x}</b><extra></extra>'
        ),
        row=2, col=1
    )
    
    # % Contango
    fig.add_trace(
        go.Bar(
            x=monthly_stats['month_name'],
            y=monthly_stats['contango_pct'],
            marker_color='#ef5350',
            name='% Contango',
            hovertemplate='<b>%{x}</b><br>Contango: %{y:.1f}%<extra></extra>'
        ),
        row=2, col=2
    )
    
    fig.update_layout(
        height=700,
        showlegend=False,
        plot_bgcolor='rgba(30, 33, 48, 0.4)',
        paper_bgcolor='rgba(0, 0, 0, 0)',
        font=dict(color='#a0a8c5', family='Arial')
    )
    
    fig.update_xaxes(gridcolor='rgba(139, 146, 176, 0.1)')
    fig.update_yaxes(gridcolor='rgba(139, 146, 176, 0.1)')
    
    return fig

def plot_vxx_simulation(sim_df):
    """Graficar simulación de rendimiento VXX"""
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Valor de la Cartera VXX', 'Retornos Diarios (%)'),
        row_heights=[0.7, 0.3],
        vertical_spacing=0.1
    )
    
    # Valor de la cartera
    fig.add_trace(
        go.Scatter(
            x=sim_df['date'],
            y=sim_df['value'],
            mode='lines',
            name='Valor de Cartera',
            line=dict(color='#ef5350', width=3),
            fill='tozeroy',
            fillcolor='rgba(239, 83, 80, 0.1)',
            hovertemplate='<b>Fecha:</b> %{x}<br><b>Valor:</b> $%{y:,.2f}<extra></extra>'
        ),
        row=1, col=1
    )
    
    # Retornos diarios
    colors = ['#26a69a' if r >= 0 else '#ef5350' for r in sim_df['return']]
    fig.add_trace(
        go.Bar(
            x=sim_df['date'],
            y=sim_df['return'],
            marker_color=colors,
            name='Retorno Diario',
            hovertemplate='<b>Fecha:</b> %{x}<br><b>Retorno:</b> %{y:.2f}%<extra></extra>'
        ),
        row=2, col=1
    )
    
    fig.update_layout(
        height=600,
        showlegend=False,
        plot_bgcolor='rgba(30, 33, 48, 0.4)',
        paper_bgcolor='rgba(0, 0, 0, 0)',
        font=dict(color='#a0a8c5', family='Arial')
    )
    
    fig.update_xaxes(gridcolor='rgba(139, 146, 176, 0.1)')
    fig.update_yaxes(gridcolor='rgba(139, 146, 176, 0.1)')
    
    return fig

def plot_historical_spread(stats_df, window=20):
    """Graficar spread histórico M1-M2 con promedio móvil - estilo mejorado"""
    fig = go.Figure()
    
    stats_df = stats_df.sort_values('date')
    stats_df['spread_ma'] = stats_df['spread'].rolling(window=window).mean()
    
    # Color basado en contango/backwardation
    colors = ['#ef5350' if s == 'Contango' else '#26a69a' for s in stats_df['state']]
    
    # Añadir scatter con efecto gradiente
    fig.add_trace(go.Scatter(
        x=stats_df['date'],
        y=stats_df['spread'],
        mode='markers',
        name='Spread M1-M2',
        marker=dict(
            size=4,
            color=colors,
            opacity=0.5,
            line=dict(width=0)
        ),
        hovertemplate='<b>Fecha:</b> %{x}<br><b>Spread:</b> %{y:.2f}<extra></extra>'
    ))
    
    # Añadir promedio móvil con estilo gradiente
    fig.add_trace(go.Scatter(
        x=stats_df['date'],
        y=stats_df['spread_ma'],
        mode='lines',
        name=f'Media Móvil {window} días',
        line=dict(
            color='#ffb74d',
            width=3,
            shape='spline'
        ),
        hovertemplate='<b>Fecha:</b> %{x}<br><b>MA:</b> %{y:.2f}<extra></extra>'
    ))
    
    fig.add_hline(y=0, line_dash="solid", line_color="rgba(139, 146, 176, 0.3)", line_width=1)
    
    fig.update_layout(
        title="Spread M1-M2 en el Tiempo",
        xaxis_title="Fecha",
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
    """Graficar distribución de contango vs backwardation - estilo mejorado"""
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
            hovertemplate='<b>%{x}</b><br>Conteo: %{y}<br>Porcentaje: %{customdata:.1f}%<extra></extra>',
            customdata=[(v/state_counts.sum()*100) for v in state_counts.values]
        )
    ])
    
    fig.update_layout(
        title="Distribución de Estados del Mercado",
        xaxis_title="Estado",
        yaxis_title="Número de Días",
        plot_bgcolor='rgba(30, 33, 48, 0.4)',
        paper_bgcolor='rgba(0, 0, 0, 0)',
        font=dict(color='#a0a8c5', family='Arial'),
        xaxis=dict(gridcolor='rgba(139, 146, 176, 0.1)'),
        yaxis=dict(gridcolor='rgba(139, 146, 176, 0.1)', showgrid=True),
        height=380
    )
    
    return fig

def plot_spread_distribution(stats_df):
    """Graficar distribución de spreads M1-M2 - estilo mejorado"""
    fig = go.Figure()
    
    fig.add_trace(go.Histogram(
        x=stats_df['spread'],
        nbinsx=50,
        marker=dict(
            color='#64b5f6',
            opacity=0.8,
            line=dict(color='white', width=1)
        ),
        name='Distribución',
        hovertemplate='<b>Rango de Spread:</b> %{x}<br><b>Conteo:</b> %{y}<extra></extra>'
    ))
    
    # Añadir línea de media con anotación mejorada
    mean_spread = stats_df['spread'].mean()
    fig.add_vline(
        x=mean_spread, 
        line_dash="dash", 
        line_color="#ffb74d",
        line_width=2,
        annotation_text=f"  Media: {mean_spread:.2f}  ",
        annotation=dict(
            bgcolor="rgba(255, 183, 77, 0.2)",
            bordercolor="#ffb74d",
            borderwidth=1,
            font=dict(color="#ffb74d", size=11)
        )
    )
    
    fig.update_layout(
        title="Distribución del Spread M1-M2",
        xaxis_title="Spread (VIX2 - VIX1)",
        yaxis_title="Frecuencia",
        plot_bgcolor='rgba(30, 33, 48, 0.4)',
        paper_bgcolor='rgba(0, 0, 0, 0)',
        font=dict(color='#a0a8c5', family='Arial'),
        xaxis=dict(gridcolor='rgba(139, 146, 176, 0.1)', showgrid=True),
        yaxis=dict(gridcolor='rgba(139, 146, 176, 0.1)', showgrid=True),
        height=380
    )
    
    return fig

def plot_roll_yield_history(df, lookback_days=252):
    """Graficar historial de roll yield en el tiempo - estilo mejorado"""
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
    
    fig.add_trace(go.Scatter(
        x=ry_df['date'],
        y=ry_df['monthly'],
        mode='lines',
        name='Roll Yield Mensual %',
        line=dict(color='#ba68c8', width=3, shape='spline'),
        fill='tozeroy',
        fillcolor='rgba(186, 104, 200, 0.15)',
        hovertemplate='<b>Fecha:</b> %{x}<br><b>Mensual:</b> %{y:.2f}%<extra></extra>'
    ))
    
    fig.add_hline(y=0, line_dash="solid", line_color="rgba(139, 146, 176, 0.3)", line_width=2)
    
    fig.update_layout(
        title=f"Historial de Roll Yield (Últimos {lookback_days} Días)",
        xaxis_title="Fecha",
        yaxis_title="Roll Yield Mensual (%)",
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
    """Graficar z-score del spread M1-M2 - estilo mejorado"""
    stats_df = stats_df.sort_values('date').copy()
    stats_df['spread_mean'] = stats_df['spread'].rolling(window=window).mean()
    stats_df['spread_std'] = stats_df['spread'].rolling(window=window).std()
    stats_df['zscore'] = (stats_df['spread'] - stats_df['spread_mean']) / stats_df['spread_std']
    
    fig = go.Figure()
    
    # Color basado en magnitud del z-score con gradiente
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
        hovertemplate='<b>Fecha:</b> %{x}<br><b>Z-Score:</b> %{y:.2f}<extra></extra>'
    ))
    
    # Añadir líneas de umbral con estilo mejorado
    fig.add_hline(
        y=2, 
        line_dash="dash", 
        line_color="#ef5350", 
        line_width=2,
        opacity=0.6,
        annotation_text="  Contango Extremo (+2σ)  ",
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
        annotation_text="  Backwardation Extremo (-2σ)  ",
        annotation=dict(
            bgcolor="rgba(38, 166, 154, 0.1)",
            bordercolor="#26a69a",
            borderwidth=1,
            font=dict(color="#26a69a", size=10)
        )
    )
    fig.add_hline(y=0, line_dash="solid", line_color="rgba(139, 146, 176, 0.3)", line_width=1)
    
    fig.update_layout(
        title=f"Z-Score de Estructura de Términos (Rolling {window} días)",
        xaxis_title="Fecha",
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
    """Crear mapa de calor de calendario de estados del mercado"""
    year_data = df[df['date'].dt.year == year].copy()
    year_data['month'] = year_data['date'].dt.month
    year_data['day'] = year_data['date'].dt.day
    year_data['state_numeric'] = year_data['state'].map({'Contango': 1, 'Backwardation': -1, 'Desconocido': 0})
    
    # Crear tabla pivot
    pivot = year_data.pivot_table(values='state_numeric', index='day', columns='month', aggfunc='mean')
    
    month_names = ['Ene', 'Feb', 'Mar', 'Abr', 'May', 'Jun', 
                   'Jul', 'Ago', 'Sep', 'Oct', 'Nov', 'Dic']
    
    fig = go.Figure(data=go.Heatmap(
        z=pivot.values,
        x=month_names,
        y=pivot.index,
        colorscale=[[0, '#26a69a'], [0.5, '#666666'], [1, '#ef5350']],
        zmid=0,
        colorbar=dict(
            title="Estado",
            tickvals=[-1, 0, 1],
            ticktext=['Back', 'Neutral', 'Contango']
        ),
        hovertemplate='<b>Mes:</b> %{x}<br><b>Día:</b> %{y}<br><b>Estado:</b> %{z:.2f}<extra></extra>'
    ))
    
    fig.update_layout(
        title=f"Calendario de Estados del Mercado - {year}",
        xaxis_title="Mes",
        yaxis_title="Día",
        plot_bgcolor='rgba(30, 33, 48, 0.4)',
        paper_bgcolor='rgba(0, 0, 0, 0)',
        font=dict(color='#a0a8c5', family='Arial'),
        height=500
    )
    
    return fig

# Aplicación principal
def main():
    # Encabezado mejorado con gradiente e iconos
    st.markdown("""
    <div style='text-align: center; padding: 2rem 0 1rem 0;'>
        <h1 style='font-size: 3rem; margin-bottom: 0.5rem; background: linear-gradient(135deg, #64b5f6 0%, #ba68c8 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;'>
            📊 Panel de Estructura de Términos VIX
        </h1>
        <p style='font-size: 1.1rem; color: #a0a8c5; font-weight: 500; letter-spacing: 1px;'>
            Análisis Avanzado de Volatilidad e Insights de Trading
        </p>
        <div style='width: 100px; height: 3px; background: linear-gradient(90deg, #64b5f6, #ba68c8); margin: 1rem auto; border-radius: 2px;'></div>
    </div>
    """, unsafe_allow_html=True)
    
    # Cargar datos
    with st.spinner('🔄 Cargando datos VIX...'):
        df = load_data()
        stats_df = calculate_historical_stats(df)
    
    # Encabezado mejorado de barra lateral
    st.sidebar.markdown("""
    <div style='text-align: center; padding: 1.5rem 0; background: rgba(100, 181, 246, 0.1); border-radius: 15px; margin-bottom: 1rem;'>
        <h2 style='margin: 0; background: linear-gradient(135deg, #64b5f6 0%, #ba68c8 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; font-size: 1.5rem;'>
            ⚙️ Controles
        </h2>
    </div>
    """, unsafe_allow_html=True)
    
    # Selección de fecha
    st.sidebar.subheader("📅 Selección de Fecha")
    available_dates = sorted(df['data_date'].unique())
    
    max_date = available_dates[-1]
    selected_date = st.sidebar.date_input(
        "Seleccionar Fecha",
        value=max_date,
        min_value=available_dates[0].date(),
        max_value=max_date.date()
    )
    selected_date = pd.to_datetime(selected_date)
    
    # Encontrar la fecha disponible más cercana
    if selected_date not in available_dates:
        nearest_date = min(available_dates, key=lambda x: abs(x - selected_date))
        st.sidebar.warning(f"Fecha ajustada a la más cercana disponible: {nearest_date.strftime('%Y-%m-%d')}")
        selected_date = nearest_date
    
    # Fechas de comparación
    st.sidebar.markdown("---")
    st.sidebar.subheader("📊 Fechas de Comparación")
    enable_comparison = st.sidebar.checkbox("Habilitar Comparación", value=False)
    
    comparison_dates = []
    if enable_comparison:
        num_comparisons = st.sidebar.slider("Número de comparaciones", 1, 4, 2)
        for i in range(num_comparisons):
            comp_date = st.sidebar.date_input(
                f"Comparación {i+1}",
                value=(max_date - timedelta(days=30*(i+1))).date(),
                key=f"comp_{i}"
            )
            comparison_dates.append(pd.to_datetime(comp_date))
    
    # Opciones de análisis
    st.sidebar.markdown("---")
    st.sidebar.subheader("⚙️ Opciones de Análisis")
    show_spot_vix = st.sidebar.checkbox("Mostrar Línea de Referencia VIX1", value=True)
    rolling_window = st.sidebar.slider("Ventana Rolling (días)", 10, 100, 20)
    zscore_window = st.sidebar.slider("Ventana Z-Score (días)", 60, 500, 252)
    
    # Contenido principal
    current_state, slope, vix1, vix2 = calculate_market_state(df, selected_date)
    daily_ry, monthly_ry, annual_ry = calculate_roll_yield(df, selected_date)
    
    # Calcular curvatura
    date_data = df[df['data_date'] == selected_date].sort_values('days_to_expiration')
    butterfly = np.nan
    if len(date_data) >= 3:
        butterfly = date_data.iloc[0]['price'] - 2*date_data.iloc[1]['price'] + date_data.iloc[2]['price']
    
    # Calcular persistencia
    current_state_data = stats_df[stats_df['date'] <= selected_date].copy()
    if len(current_state_data) > 0:
        current_state_data = current_state_data.sort_values('date')
        current_state_data['state_change'] = current_state_data['state'] != current_state_data['state'].shift(1)
        
        # Encontrar la racha actual
        current_streak = 1
        for i in range(len(current_state_data)-1, 0, -1):
            if current_state_data.iloc[i]['state'] == current_state_data.iloc[i-1]['state']:
                current_streak += 1
            else:
                break
    else:
        current_streak = 0
    
    # Métricas principales mejoradas con tarjetas de gradiente
    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    
    with col1:
        badge_class = "badge-contango" if current_state == "Contango" else "badge-backwardation"
        badge_emoji = "🔴" if current_state == "Contango" else "🟢"
        st.markdown(f"""
        <div style='text-align: center; padding: 10px;'>
            <p style='color: #a0a8c5; font-size: 0.85rem; margin-bottom: 8px; font-weight: 500; text-transform: uppercase; letter-spacing: 0.5px;'>Estado</p>
            <div class='badge {badge_class}' style='font-size: 0.85rem;'>{badge_emoji} {current_state.upper()}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-title'>📈 VIX1</div>
            <div class='metric-value'>{vix1:.2f}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        delta_color = "#ef5350" if slope > 0 else "#26a69a" if slope < 0 else "#8b92b0"
        delta_symbol = "▲" if slope > 0 else "▼" if slope < 0 else "●"
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-title'>📊 VIX2</div>
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
                <div class='metric-title'>📅 Roll Yield</div>
                <div class='metric-value' style='color: {ry_color};'>{monthly_ry*100:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class='metric-card'>
                <div class='metric-title'>📅 Roll Yield</div>
                <div class='metric-value' style='color: #8b92b0;'>N/A</div>
            </div>
            """, unsafe_allow_html=True)
    
    with col5:
        if not pd.isna(butterfly):
            bf_color = "#64b5f6"
            st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-title'>🦋 Butterfly</div>
                <div class='metric-value' style='color: {bf_color}; font-size: 1.5rem;'>{butterfly:.2f}</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class='metric-card'>
                <div class='metric-title'>🦋 Butterfly</div>
                <div class='metric-value' style='color: #8b92b0; font-size: 1.5rem;'>N/A</div>
            </div>
            """, unsafe_allow_html=True)
    
    with col6:
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-title'>⏱️ Racha</div>
            <div class='metric-value' style='font-size: 1.5rem;'>{current_streak}</div>
            <div style='color: #a0a8c5; font-size: 0.75rem; margin-top: 4px;'>días</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Encabezado de sección mejorado
    st.markdown("""
    <div style='margin-bottom: 1.5rem;'>
        <h2 style='display: inline-block; margin: 0; font-size: 1.8rem;'>📈 Curva de Estructura de Términos</h2>
        <p style='color: #a0a8c5; margin-top: 0.5rem; font-size: 0.95rem;'>Visualiza los precios de futuros VIX a través de diferentes vencimientos</p>
    </div>
    """, unsafe_allow_html=True)
    
    fig_ts = plot_term_structure(df, selected_date, comparison_dates if enable_comparison else None, show_spot_vix)
    st.plotly_chart(fig_ts, use_container_width=True)
    
    # Sección de pestañas mejorada
    st.markdown("<br>", unsafe_allow_html=True)
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
        "📊 Spread Histórico",
        "🎯 Análisis de Distribución",
        "📉 Roll Yield",
        "🔬 Análisis Z-Score",
        "📅 Vista de Calendario",
        "🌡️ Estacionalidad",
        "🚨 Días Extremos",
        "💰 Simulador VXX"
    ])
    
    with tab1:
        st.subheader("Spread M1-M2 en el Tiempo")
        fig_spread = plot_historical_spread(stats_df, window=rolling_window)
        st.plotly_chart(fig_spread, use_container_width=True)
        
        # Estadísticas
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Spread Medio", f"{stats_df['spread'].mean():.2f}")
        with col2:
            st.metric("Desv. Estándar", f"{stats_df['spread'].std():.2f}")
        with col3:
            contango_pct = (stats_df['state'] == 'Contango').sum() / len(stats_df) * 100
            st.metric("% en Contango", f"{contango_pct:.1f}%")
        with col4:
            backwardation_pct = (stats_df['state'] == 'Backwardation').sum() / len(stats_df) * 100
            st.metric("% en Backwardation", f"{backwardation_pct:.1f}%")
    
    with tab2:
        col1, col2 = st.columns(2)
        with col1:
            fig_state_dist = plot_market_state_distribution(stats_df)
            st.plotly_chart(fig_state_dist, use_container_width=True)
        with col2:
            fig_spread_dist = plot_spread_distribution(stats_df)
            st.plotly_chart(fig_spread_dist, use_container_width=True)
        
        # Análisis de percentiles
        st.markdown("### 📊 Análisis de Percentiles")
        current_spread = vix2 - vix1
        percentile = stats.percentileofscore(stats_df['spread'].dropna(), current_spread)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Spread Actual", f"{current_spread:.2f}")
        with col2:
            st.metric("Rango Percentil", f"{percentile:.1f}%")
        with col3:
            if percentile > 90:
                st.error("🔴 Contango Extremadamente Alto")
            elif percentile < 10:
                st.success("🟢 Backwardation Extremo")
            else:
                st.info("🔵 Rango Normal")
    
    with tab3:
        fig_roll = plot_roll_yield_history(df, lookback_days=min(252, len(stats_df)))
        st.plotly_chart(fig_roll, use_container_width=True)
        
        st.markdown("""
        ### 💡 Explicación de Roll Yield
        
        **Roll Yield Positivo (Contango)**: Los futuros VIX cotizan a precios más altos que el VIX spot. 
        Esto crea un **viento en contra** para ETFs largos en VIX (VXX, UVXY), ya que pierden valor al rodar contratos.
        
        **Roll Yield Negativo (Backwardation)**: Los futuros VIX cotizan a precios más bajos que el VIX spot.
        Esto crea un **viento a favor** para ETFs largos en VIX, ya que ganan valor al rodar contratos.
        
        **Para Traders de ETF VIX**: 
        - Alto contango (roll yield positivo) = La decadencia se acelera para posiciones largas
        - Backwardation (roll yield negativo) = Posiciones largas se benefician del roll
        """)
        
        # Tabla de resumen de roll yield
        if daily_ry is not None:
            st.markdown("### 📋 Resumen Actual de Roll Yield")
            roll_summary = pd.DataFrame({
                'Período': ['Diario', 'Mensual', 'Anual'],
                'Roll Yield (%)': [daily_ry*100, monthly_ry*100, annual_ry*100],
                'Impacto en VXX': [
                    'Mínimo' if abs(daily_ry) < 0.01 else 'Moderado' if abs(daily_ry) < 0.03 else 'Severo',
                    'Mínimo' if abs(monthly_ry) < 0.05 else 'Moderado' if abs(monthly_ry) < 0.15 else 'Severo',
                    'Mínimo' if abs(annual_ry) < 0.50 else 'Moderado' if abs(annual_ry) < 1.5 else 'Severo'
                ]
            })
            st.dataframe(roll_summary, use_container_width=True)
    
    with tab4:
        fig_zscore = plot_zscore_analysis(stats_df, window=zscore_window)
        st.plotly_chart(fig_zscore, use_container_width=True)
        
        st.markdown("""
        ### 🎯 Interpretación del Z-Score
        
        El z-score mide qué tan inusual es la estructura de términos actual comparada con las normas históricas:
        
        - **Z-Score > +2**: Contango extremadamente pronunciado (evento raro, >percentil 95)
        - **Z-Score entre -2 y +2**: Rango normal
        - **Z-Score < -2**: Backwardation extremo (evento raro, <percentil 5)
        
        **Implicaciones de Trading**:
        - Lecturas extremas a menudo preceden a reversión a la media
        - Z-score > +2: Considerar estrategias cortas en volatilidad
        - Z-score < -2: A menudo ocurre durante estrés de mercado/crashes
        """)
        
        # Z-score actual
        current_spread = vix2 - vix1
        recent_stats = stats_df.tail(zscore_window)
        if len(recent_stats) >= zscore_window:
            mean_spread = recent_stats['spread'].mean()
            std_spread = recent_stats['spread'].std()
            current_zscore = (current_spread - mean_spread) / std_spread
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Z-Score Actual", f"{current_zscore:.2f}")
            with col2:
                st.metric(f"Media {zscore_window} Días", f"{mean_spread:.2f}")
            with col3:
                st.metric(f"Desv. Est. {zscore_window} Días", f"{std_spread:.2f}")
    
    with tab5:
        st.subheader("Mapa de Calor de Calendario")
        selected_year = st.selectbox(
            "Seleccionar Año",
            sorted(stats_df['date'].dt.year.unique(), reverse=True)
        )
        
        fig_calendar = plot_heatmap_calendar(stats_df, selected_year)
        st.plotly_chart(fig_calendar, use_container_width=True)
        
        # Resumen del año
        year_stats = stats_df[stats_df['date'].dt.year == selected_year]
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Días de Trading", len(year_stats))
        with col2:
            contango_days = (year_stats['state'] == 'Contango').sum()
            st.metric("Días Contango", contango_days)
        with col3:
            back_days = (year_stats['state'] == 'Backwardation').sum()
            st.metric("Días Backwardation", back_days)
        with col4:
            avg_spread = year_stats['spread'].mean()
            st.metric("Spread Promedio", f"{avg_spread:.2f}")
    
    with tab6:
        st.subheader("🌡️ Análisis de Estacionalidad")
        st.markdown("""
        Examina patrones mensuales en la estructura de términos VIX. 
        Ciertos meses tienden históricamente hacia mayor contango o backwardation.
        """)
        
        monthly_stats = calculate_seasonality(stats_df)
        fig_season = plot_seasonality(monthly_stats)
        st.plotly_chart(fig_season, use_container_width=True)
        
        # Tabla de resumen mensual
        st.markdown("### 📋 Resumen Mensual")
        display_stats = monthly_stats[['month_name', 'spread_mean', 'vix1_mean', 'contango_pct']].copy()
        display_stats.columns = ['Mes', 'Spread Promedio', 'VIX1 Promedio', '% Contango']
        st.dataframe(display_stats, use_container_width=True)
        
        # Insights
        best_month = monthly_stats.loc[monthly_stats['spread_mean'].idxmax(), 'month_name']
        worst_month = monthly_stats.loc[monthly_stats['spread_mean'].idxmin(), 'month_name']
        
        st.markdown(f"""
        ### 📊 Insights Clave
        - **Mes con Mayor Contango**: {best_month}
        - **Mes con Menor Contango**: {worst_month}
        - El VIX tiende a mostrar patrones estacionales relacionados con eventos del mercado
        - Octubre históricamente muestra mayor volatilidad (crashes famosos)
        """)
    
    with tab7:
        st.subheader("🚨 Días Extremos en la Historia")
        st.markdown("Explora los días más extremos de contango, backwardation y cambios de spread.")
        
        top_contango, top_backwardation, top_changes = find_extreme_days(stats_df, n=10)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("#### 🔴 Top 10 Contango")
            display_contango = top_contango[['date', 'spread', 'vix1']].copy()
            display_contango['date'] = display_contango['date'].dt.strftime('%Y-%m-%d')
            display_contango.columns = ['Fecha', 'Spread', 'VIX1']
            st.dataframe(display_contango, use_container_width=True, hide_index=True)
        
        with col2:
            st.markdown("#### 🟢 Top 10 Backwardation")
            display_back = top_backwardation[['date', 'spread', 'vix1']].copy()
            display_back['date'] = display_back['date'].dt.strftime('%Y-%m-%d')
            display_back.columns = ['Fecha', 'Spread', 'VIX1']
            st.dataframe(display_back, use_container_width=True, hide_index=True)
        
        with col3:
            st.markdown("#### ⚡ Top 10 Cambios")
            display_changes = top_changes[['date', 'spread_change', 'vix1']].copy()
            display_changes['date'] = display_changes['date'].dt.strftime('%Y-%m-%d')
            display_changes.columns = ['Fecha', 'Cambio', 'VIX1']
            st.dataframe(display_changes, use_container_width=True, hide_index=True)
        
        # Saltar a fecha extrema
        st.markdown("### 🔍 Explorar Fecha Extrema")
        extreme_type = st.selectbox(
            "Tipo de Extremo",
            ["Contango Extremo", "Backwardation Extremo", "Mayor Cambio"]
        )
        
        if extreme_type == "Contango Extremo":
            selected_extreme = st.selectbox(
                "Seleccionar Fecha",
                top_contango['date'].dt.strftime('%Y-%m-%d').tolist()
            )
        elif extreme_type == "Backwardation Extremo":
            selected_extreme = st.selectbox(
                "Seleccionar Fecha",
                top_backwardation['date'].dt.strftime('%Y-%m-%d').tolist()
            )
        else:
            selected_extreme = st.selectbox(
                "Seleccionar Fecha",
                top_changes['date'].dt.strftime('%Y-%m-%d').tolist()
            )
        
        if st.button("🚀 Ir a Esta Fecha"):
            st.info(f"Actualiza la fecha en la barra lateral a: {selected_extreme}")
    
    with tab8:
        st.subheader("💰 Simulador de Rendimiento VXX")
        st.markdown("""
        Simula cómo habría funcionado una inversión en VXX usando el carry diario histórico de tu base de datos.
        VXX típicamente pierde valor en contango y gana en backwardation.
        """)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            sim_start = st.date_input(
                "Fecha de Inicio",
                value=(max_date - timedelta(days=365)).date(),
                min_value=available_dates[0].date(),
                max_value=max_date.date(),
                key="sim_start"
            )
        
        with col2:
            sim_end = st.date_input(
                "Fecha de Fin",
                value=max_date.date(),
                min_value=available_dates[0].date(),
                max_value=max_date.date(),
                key="sim_end"
            )
        
        with col3:
            initial_investment = st.number_input(
                "Inversión Inicial ($)",
                min_value=1000,
                max_value=1000000,
                value=10000,
                step=1000
            )
        
        if st.button("▶️ Ejecutar Simulación"):
            with st.spinner("Calculando rendimiento..."):
                sim_df = simulate_vxx_performance(
                    df, 
                    pd.to_datetime(sim_start), 
                    pd.to_datetime(sim_end),
                    initial_investment
                )
                
                if len(sim_df) > 0:
                    # Gráfico
                    fig_sim = plot_vxx_simulation(sim_df)
                    st.plotly_chart(fig_sim, use_container_width=True)
                    
                    # Métricas de rendimiento
                    final_value = sim_df['value'].iloc[-1]
                    total_return = ((final_value / initial_investment) - 1) * 100
                    max_value = sim_df['value'].max()
                    min_value = sim_df['value'].min()
                    max_drawdown = ((min_value / max_value) - 1) * 100
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric(
                            "Valor Final",
                            f"${final_value:,.2f}",
                            f"{total_return:+.2f}%"
                        )
                    
                    with col2:
                        st.metric(
                            "Retorno Total",
                            f"{total_return:.2f}%"
                        )
                    
                    with col3:
                        st.metric(
                            "Valor Máximo",
                            f"${max_value:,.2f}"
                        )
                    
                    with col4:
                        st.metric(
                            "Caída Máxima",
                            f"{max_drawdown:.2f}%"
                        )
                    
                    st.markdown(f"""
                    ### 📊 Resumen de Simulación
                    
                    - **Período**: {sim_start} a {sim_end}
                    - **Días**: {len(sim_df) - 1} días de trading
                    - **Inversión Inicial**: ${initial_investment:,.2f}
                    - **Valor Final**: ${final_value:,.2f}
                    - **Ganancia/Pérdida**: ${final_value - initial_investment:,.2f}
                    
                    ⚠️ **Nota**: Esta simulación usa el carry diario histórico como proxy para el rendimiento de VXX.
                    Los resultados reales pueden variar debido a factores como rebalanceo de ETF, costos de gestión y slippage.
                    """)
                else:
                    st.warning("No hay suficientes datos para el rango de fechas seleccionado.")
    
    # Exportar sección de datos
    st.markdown("---")
    st.subheader("📥 Exportar Datos")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Exportar estructura de términos actual
        current_ts = df[df['data_date'] == selected_date].sort_values('days_to_expiration')
        csv_ts = current_ts.to_csv(index=False)
        st.download_button(
            label="Descargar Estructura Actual",
            data=csv_ts,
            file_name=f"vix_ts_{selected_date.strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )
    
    with col2:
        # Exportar estadísticas históricas
        csv_stats = stats_df.to_csv(index=False)
        st.download_button(
            label="Descargar Estadísticas Históricas",
            data=csv_stats,
            file_name="vix_estadisticas_historicas.csv",
            mime="text/csv"
        )
    
    with col3:
        # Exportar rango filtrado
        date_range = st.sidebar.date_input(
            "Rango de Exportación",
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
                label="Descargar Rango Filtrado",
                data=csv_filtered,
                file_name=f"vix_data_{date_range[0]}_{date_range[1]}.csv",
                mime="text/csv"
            )
    
    # Pie de página
    st.markdown("<br><br><br>", unsafe_allow_html=True)
    st.markdown("""
    <div class='footer'>
        Hecho con ❤️ por <strong>@Gsnchez</strong> | <a href='http://bquantfinance.com' target='_blank' style='color: #64b5f6; text-decoration: none;'>bquantfinance.com</a>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
