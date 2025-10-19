import streamlit as st
import pandas as pd
import plotly.graph_objects as go

#функция визцализации предсказаний и истинных значений
#y_test - истинные значения
#y_pred - предсказанные значения
#title - добавление к названию

def charts_residuals_plotly(y_test, y_pred, title=None):
    
    df = pd.DataFrame(y_test)
    
    df = df.join(y_pred)
    df.columns = ['y_true', 'y_pred']

    df['residuals'] = df['y_true'] - df['y_pred']

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=df['y_true'],
            y=df['y_pred'],
            mode='markers',
            name='Предсказания'
        )
    )

    min_val = min(df['y_true'].min(), df['y_pred'].min())
    max_val = max(df['y_true'].max(), df['y_pred'].max())
    
    fig.add_trace(
        go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode='lines',
            line=dict(dash='dash', color='black'),
            name='линия идеальных предсказаний'
        )
    )
        
   
    fig.update_layout(
        title={
            'text': f'Сравнение предсказаний с истинными значениями: {title}',
            'x': 0
        },
        xaxis_title='Истинные значения',
        yaxis_title='Предсказанные значения',
        showlegend=True
    )

    return fig