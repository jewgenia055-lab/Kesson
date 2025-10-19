import streamlit as st
import pandas as pd
import numpy as np

from utils.kesson_model import kesson_model #импорт сети
import utils.constants as const  #импорт констрант
from utils.optimizer import optimizer_kesson_model  #импорт оптимизатора
import utils.calculation as calc #импорт функции форматирования жесткости

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import joblib

import random

import plotly.graph_objects as go


st.title("Подбор толщин силовых элементов кессона")

st.markdown("""Для подбора толщин силовых элементов кессона необходимо ввести:
1. Целевую массу (кг) - желаемую массу конструкции в килограммах.   
2. Стартовые толщины (мм) - предположение по толщинам силовых элементов, с которого начнется подбор.   

**Техническое ограничение:**     
Для корректной работы оптимизатора начальные значения толщин должны находиться в рабочем диапазоне (0.8–2.5 мм). Выход за эти пределы может привести к некорректным прогнозам жесткости.

""")

st.image("pictures/optim.png", caption="Cиловые элементы кессона", width=450)

with st.form("form", width='content'):
    M_target_original = st.number_input("Масса, кг", value="min", min_value=0.)
    thikness_skin_top = st.number_input("Предполагаемая толщина верхней обшивки, мм", value="min", min_value=0.)
    thikness_skin_bot = st.number_input("Предполагаемая толщина нижней обшивки, мм", value="min", min_value=0.)
    thikness_long_front = st.number_input("Предполагаемая толщина стенки переднего лонжерона, мм", value="min", min_value=0.)
    thikness_long_back = st.number_input("Предполагаемая толщина стенки заднего лонжерона, мм", value="min", min_value=0.)
    
    submitted = st.form_submit_button("Подобрать")


if submitted:
    #сохранение результатов в session_state
    st.session_state.initial_params = {
        'target_mass': M_target_original,
        'skin_top': thikness_skin_top,
        'skin_bot': thikness_skin_bot,
        'long_front': thikness_long_front,
        'long_back': thikness_long_back        
    }


    #инициализация сети
    model = kesson_model().load()
    
    #обработка введенных значений
    initial_params = st.session_state.initial_params 
    
    #нормализация заданной массы
    M_target = initial_params['target_mass'] / 1000  #перевод в тонны
    
    #нормализация заданной массы
    M_target_np = np.array([[M_target, 0]])
    M_target_scaled = model.scaler_y.transform(M_target_np)[0, 0]
    
    #точка начала оптимизации
    initial_point =  st.session_state.initial_point
    #заполнение списка точки начала оптимизации
    initial_point = [
        initial_params['skin_top'],
        initial_params['skin_bot'],
        initial_params['long_front'],
        initial_params['long_back']
    ]

    #инициализация оптимизатора
    optimizer = optimizer_kesson_model(model=model)
    
    #подбор толщин силовых элементов
    results = st.session_state.results
    
    results = optimizer.optimize(
        x_initial=initial_point,
        target_mass_scaled=M_target_scaled
    )

    continuous_stiffness_latex = calc.format_to_latex(results['continuous_stiffness'])          

    st.subheader("Вычисленные параметры модели")

    results_data = { 
        'Масса конструкции, кг' :  round(results['discrete_mass'] * 1000, 2),
        'Жесткость конструкции, Нмм²/рад' :  results['discrete_stiffness'],
        
        'Толщина верхней обшивки, мм' : results['discrete_params'][0],
        'Толщина нижней обшивки, мм' : results['discrete_params'][1],
        'Толщина стенки переднего лонжерона, мм' : results['discrete_params'][2],
        'Толщина стенки заднего лонжерона, мм' : results['discrete_params'][3]
    }

    df = pd.DataFrame.from_dict(
        results_data,
        orient='index',
        columns=['Результаты']
    )
    st.data_editor(
        df,
        column_config={
            '_index': st.column_config.Column("Параметры"),  # индексный столбец
            'Результаты': st.column_config.Column(width="medium")
            },
        use_container_width=False
        )

  



        










