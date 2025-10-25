[![Python](https://img.shields.io/badge/Python-3.12+-3776AB?logo=python&logoColor=white)](https://python.org)
[![Pandas](https://img.shields.io/badge/Pandas-2.2+-150458?logo=pandas&logoColor=white)](https://pandas.pydata.org)
[![NumPy](https://img.shields.io/badge/NumPy-1.26+-013243?logo=numpy&logoColor=white)](https://numpy.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.5+-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.6+-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org)
[![phik](https://img.shields.io/badge/phik-0.12+-blue?logo=databricks&logoColor=white)](https://github.com/KaveIO/PhiK)
[![Plotly](https://img.shields.io/badge/Plotly-5.24+-3F4F75?logo=plotly&logoColor=white)](https://plotly.com)
[![Docker](https://img.shields.io/badge/Docker-28.5+-2496ED?logo=docker&logoColor=white)](https://docker.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.50+-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io)


# Kesson

Оптимизатор, подбирающий толщины обшивок и лонжеронов кессона по многокритериальным ограничениям.

### О проекте

На определенном этапе проектирования, когда утверждены внешние обводы и массы агрегатов, необходимо подбирать поперечные сечения и толщины силовых элементов конструкции. Для этой цели создаются метамодели, по которым проводится подбор оптимальных параметров конструкции на основе выбранных критериев.  
В данном проекте критериями являются:
1. минимальное отклонение от заданной массы (утвержденной)
2. максимальное значение изгибной жесткости.
3. учет технологических требований к толщинам


### Данные

**Источник данных :** В данном проекте исходные данные синтетические, результаты расчета Femap


### Запуск тетради проекта

**Важно :** Перед запуском тетради убедитесь, что структура проекта сохранена.

## Приложение

**Приложение Streamlit :** https://keappn-pu78x2bvkhfs5xi8hshycy.streamlit.app/

## Структура проекта

***Файлы проекта***
| Файл | Назначение |
|------------|------------|
| `.dockerignore` | Игнорируемые файлы Docker |
| `Dockerfile` | Файл Docker |
| `.gitignore` | Игнорируемые файлы |
| `app.py` | Streamlit приложение |
| `requirements.txt` | Зависимости |
| `README.md` | Этот файл |

***Данные : data/***
| Файл | Назначение |
|------------|------------|
| `all_prop.xlsx` | Массы и толщины моделей |
| `all_solve.xlsx` | Результаты расчета моделей |
| `fem_characteristics_valid.xlsx` | Массы и толщины валидационных моделей |
| `geom_param.xlsx` | Длины секций |
| `predictions_test.csv` | Предсказания на тестовых данных |
| `results_valid.xlsx` | Результаты расчета валидационных моделей |
| `valid_1_m_1.csv` | Подбор оптимизатора 1 этап масса 1 кг |
| `valid_1_m_100.csv` | Подбор оптимизатора 1 этап масса 100 кг |
| `valid_1_m_100_fem.csv` | Валидация оптимизатора 1 этап масса 100 кг |
| `valid_1_m_1_fem.csv` | Валидация оптимизатора 1 этап масса 1 кг |
| `valid_1_m_8.csv` | Подбор оптимизатора 1 этап масса 8 кг |
| `valid_1_m_8_fem.csv` | Валидация оптимизатора 1 этап масса 8 кг |
| `valid_2_max_delta.csv` | Подбор оптимизатора 2 этап максимальные толщины |
| `valid_2_max_delta_fem.csv` | Валидация оптимизатора 2 этап максимальные толщины |
| `valid_2_min_delta.csv` | Подбор оптимизатора 2 этап минимальные толщины |
| `valid_2_min_delta_fem.csv` | Валидация оптимизатора 2 этап минимальные толщины |
| `valid_2_random_delta.csv` | Подбор оптимизатора 2 этап толщины вне рабочего диапазона |
| `valid_2_random_delta_fem.csv` | Валидация оптимизатора 2 этап толщины вне рабочего диапазона |
| `y_test.csv` | Тестовые данные |

***Метамодель : model/***
| Файл | Назначение |
|------------|------------|
| `kesson_model_nn_model_config.joblib` | Параметры метамодели |
| `kesson_model_nn_model_weights.pth` | Веса метамодели |

***Тетрадь проекта : notebook/***
| Файл | Назначение |
|------------|------------|
| `Kesson.ipynb` | Jupyter ноутбук проекта |

***Страницы приложения : pages/***

| Файл | Назначение |
|------------|------------|
| `conclusions.py` | Выводы |
| `metamodel.py` | Метамодель кессона |
| `model_fem.py` | Модель кессона |
| `optimiz_desc.py` | Модель оптимизатора |
| `optimization.py` | Подбор толщин |
| `project.py` | Описание проекта |
| `validation_optimiz.py` | Валидация оптимизатора |
| `validation_optimiz_1.py` | 1 этап валидации |
| `validation_optimiz_2.py` | 2 этап валидации |
| `сriterion_optimiz.py` | Критерии оптимизациии |

***Изображения : pictures/***
| Файл | Назначение |
|------------|------------|
| `deform.gif` | Деформации кессона |
| `elem.png` | Силовые элементы конструкции |
| `kesson.png` | FEM модель кессона |
| `optim.png` | Силовые элементы для которых подбираются толщины |
| `section.png` | Секции кессона |
| `val_1.png` | Валидационная модель 1 этап масса 1 кг |
| `val_100.png` | Валидационная модель 1 этап масса 100 кг |
| `val_8.png` | Валидационная модель 1 этап масса 8 кг |
| `val_max.png` | Валидационная модель 2 этап максимальные толщины |
| `val_min.png` | Валидационная модель 2 этап минимальные толщины |
| `val_random.png` | Валидационная модель 2 этап толщины вне рабочего диапазона |

***Функции приложенния : utils/***
| Файл | Назначение |
|------------|------------|
| `calculations.py` | Расчетные функции |
| `constants.py` | Констранты |
| `kesson_model.py` | Обертка метамодели  |
| `optimizer.py` | Обертка оптимизатора |
| `visualization.py` | Функции визуализации|


## Библитотеки и инструменты разработки

**Инструменты разработки**

- VSCode
- Jupyter Notebook
- Python 3.12.3
- Docker 28.5.1


**Библиотеки**
- Numpy 1.26.4
- Pandas 2.2.2
- Scipy 1.13.1
- phik 0.12.4
- Pytorch 2.6.0
- Scikit-learn 1.5.1
- joblib 1.4.2
- Plotly 5.24.1

**Визуализация**
- Streamlit 1.50.0

**Запуск**

```
#клонирование репозитория
git clone https://github.com/jewgenia055-lab/Kesson.git

#переход в директорию проекта
cd Kesson

#сборка Docker образа
docker build -t kesson . 

#запуск контейнера
docker run -p 8501:8501 kesson

#ссылка открытия в браузере
http://localhost:8501

```

