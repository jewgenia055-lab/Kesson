import streamlit as st
import pandas as pd
import os


#инициализация session_state
if 'initial_params' not in st.session_state:
    #значения по умолчанию
    st.session_state.initial_params = {} #словарь введенных параметров
if 'initial_point' not in st.session_state:
    st.session_state.initial_point = []   #список введеных толщин
if 'results' not in st.session_state:
    st.session_state.results = {} #словарь результатов оптимизации


                #Страницы проекта
pages = {
    "Введение": [
        st.Page("pages/project.py", title="Описание проекта"),
        st.Page("pages/model_fem.py", title="Модель кессона"),
        st.Page("pages/сriterion_optimiz.py", title="Критерии оптимизациии"),
    ],
    
    "Подбор толщин" : [
        st.Page("pages/optimization.py", title="Подбор толщин")
    ],
    
    "Создание оптимизатора" : [
        st.Page("pages/metamodel.py", title="Метамодель кессона"),
        st.Page("pages/optimiz_desc.py", title="Модель оптимизатора")
    ],
    
    "Валидация оптимизатора" : [
        st.Page("pages/validation_optimiz.py", title="Валидация оптимизатора"),
        st.Page("pages/validation_optimiz_1.py", title="1 этап валидации"),
        st.Page("pages/validation_optimiz_2.py", title="2 этап валидации"),
    ],


    "Выводы" : [
        st.Page("pages/conclusions.py", title="Выводы")
    ]
    
}

pg = st.navigation(pages)
pg.run()