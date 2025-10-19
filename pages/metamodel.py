import streamlit as st
import pandas as pd
import utils.visualization as vis #импорт функции визуализации остатков


st.title("Метамодель кессона")

st.markdown("""

Метамодель кессона - нейроннная сеть, принимающая на вход толщины силовых элементов и предсказывающая массу и жесткость конструкции.

""")



tab1, tab2, tab3 = st.tabs([
    "Архитектура метамодели",
    "Обучение метамодели",
    "Проверка на тестовых данных"
])


with tab1:

    st.subheader("Архитектура метамодели")
    
    st.markdown("""
    
    Описание архитектуры:
    
    Тип сети - **Полносвязный**
    
    1. Первый слой:  5 нейронов (4 - **входные** признаки, 1 - смещение)
    2. Второй слой: 64 нейрона (скрытый слой)  
       *  функция активации *ReLU*
       *  прореживание *Dropuot* с вероятностью 0.05    
    3. Третий слой: 32 нейрона (скрытый слой)
       * функция активации *Hardtanh*
       * прореживание *Dropuot* с вероятностью 0.05
    4. Выходной слой: 2 нейрона (1 - прогноз массы, 1 - прогноз жесткости)
    
    
    Особенности архитектуры:
    * оптимизатор - алгоритм *Adam*
    * ограничения переобучения:
        1. параметр алгорима *Adam* (L2 реругялизация : weight_decay)
        2. прореживание *Dropuot*
    * функция ошибки - *MSE*
    
    
    """)
    with st.expander("Код метамодели"):
        code_1 = '''#Обертка для всей сети
#включает в себя всю логику:
#масштабирование, обучение, предсказание, оценку
#сохранение сети

class kesson_model:
 
#Инициализация конвейера
    def __init__(
        self,
        input_size=None, #число входных нейронов
        output_size=2, #число выходных нейронов
        neural_layer_1=64, #число нейронов в первом слое
        neural_layer_2=32, #число нейронов во втором слое
        dropout_1=0.05,
        dropout_2=0.05, #дропаут для второго слоя
        epochs=200, #число эпох обучения
        lr=0.001, #скорость обучения
        weight_decay=1e-5, #параметр L2 регуляризации
        
        batch_size=64, #число батчей
        random_state=RANDOM_STATE #константа фиксирующая воспроизводимость
        
    ):  

        #фиксация случайностей
        self.seed = random_state
        random.seed(random_state) 
        np.random.seed(random_state)
        torch.manual_seed(random_state) 

        self.model = None
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        self.neural_layer_1 = neural_layer_1
        self.neural_layer_2 = neural_layer_2
        self.dropout1 = dropout_1  
        self.dropout2 = dropout_2  
        self.input_size = input_size
        self.output_size = output_size
        self.epochs = epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
       
        
        self.criterion = nn.MSELoss() # функция ошибки
        self.train_losses = [] # хранение истории

#Функция масштабирования train
#X_train - входные призаки в исходном масштабе
#y_train - целевые признаки в исходном масштабе
    def scaling(self, X_train, y_train):
        X_train_scaled = self.scaler_X.fit_transform(X_train)
        y_train_scaled = self.scaler_y.fit_transform(y_train)

        return X_train_scaled, y_train_scaled

#Конвертация данных в тензоры
#data - масштабированные данные
#requires_grad - нужны ли градиенты (False для предсказания)
    
    def convert_tensor(self, data, requires_grad=False):    
        return torch.tensor(data, dtype=torch.float32, requires_grad=requires_grad)

#Архитектура нейронной сети
    class KessonNN(nn.Module):
        def __init__(
            self, input_size, neural_layer_1, neural_layer_2,
            dropout1, dropout2
        ):
            super().__init__()
            self.shared_net = nn.Sequential(
                nn.Linear(input_size, neural_layer_1),
                nn.ReLU(),
                nn.Dropout(dropout1),
                nn.Linear(neural_layer_1, neural_layer_2),
                nn.Hardtanh(),
                nn.Dropout(dropout2)
            )
            self.mass_head = nn.Linear(neural_layer_2, 1)
            self.stiffness_head = nn.Linear(neural_layer_2, 1)

        def forward(self, x):
            x = self.shared_net(x)
            mass = self.mass_head(x)
            stiffness = self.stiffness_head(x)
            return torch.cat((mass, stiffness), dim=1)

#Обучение модели
#X_train - немасштабированные входные признаки (np)
#y_train - немасштабированные целевые признаки (np)
    
    def fit(self, X_train, y_train):

        #определение числа нейронов во входном слое
        self.input_size = X_train.shape[1]
 
        #масштабирование тренировочных признаков
        X_train_scaled, y_train_scaled = self.scaling(X_train, y_train)

        #конвертация тренировочных признаков в тензоры
        X_train_tensor = self.convert_tensor(X_train_scaled,
                                               requires_grad=True)
        y_train_tensor = self.convert_tensor(y_train_scaled,
                                               requires_grad=True)        

        #создание TensorDataset
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        #создание DataLoader
        train_loader = DataLoader(train_dataset, self.batch_size, shuffle=True)
            
        #создание сети
        self.model = self.KessonNN(
            self.input_size,
            self.neural_layer_1,
            self.neural_layer_2,
            self.dropout1,
            self.dropout2
        )

        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )
     
        #сброс истории перед новым обучением
        self.train_losses = []


        for i_epoch in range(1, self.epochs + 1):
            self.model.train()
            running_loss = 0.0

            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            epoch_loss = running_loss / len(train_loader)
            self.train_losses.append(epoch_loss) #loss для истории
            
            if (i_epoch) % 20 == 0:
                print(f'Epoch [{i_epoch}/{self.epochs}], Loss: {epoch_loss:.6f}')


        #возвращение self для цепочки вызовов
        return self


#Предсказания
#X - входные признаки в исходном масштабе (np)

#predictions_scaling - вывод предсказаний в исходном или в масштабе
   
    def predict(self, X, predictions_scaling=False):
        
        #масштабирование входных признаков
        X_scaled = self.scaler_X.transform(X)
         
        #конвертация входных признаков в тензор
        X_tensor = self.convert_tensor(X_scaled)

        #перевод в режим оценки
        self.model.eval()

        #предсказание
        with torch.no_grad():
            predictions = self.model(X_tensor)

        if predictions_scaling:
            #перевод предсказаний в исходный масштаб
            pred = self.scaler_y.inverse_transform(predictions.numpy())
        else:
            pred = predictions

        return pred
 
    def predict_with_grad(self, X_tensor):

        predictions = self.model(X_tensor)
        return predictions  

    
#Оценка предсказаний
#pred - предсказания в исходный масштабе
    
    def score(self, pred, y):

        self.model.eval()

        #масштабирование целевых признаков валидационных данных
        y_scaled = self.scaler_y.transform(y)
        y_tensor = self.convert_tensor(y_scaled)

        #масштабирование предсказаний
        pred_scaled = self.scaler_y.transform(pred)
        pred_tensor = self.convert_tensor(pred_scaled)

        test_loss = self.criterion(pred_tensor, y_tensor).item()
            
        r2_mass = r2_score(y[:, 0], pred[:, 0])
        r2_stiffness = r2_score(y[:, 1], pred[:, 1])

        print(f"MSE = {test_loss:.4f}")
        print(f"R2 масса = {r2_mass:.4f}")
        print(f"R2 жесткость = {r2_stiffness:.4f}")

        return {'loss' : test_loss,
                'r2_mass' : r2_mass,
                'r2_stiffness' : r2_stiffness}

#Функция вызова параметров
    def parameters(self):         
        return self.model.parameters()

#Функция перевода в режим предсказания    
    def eval(self):
        self.model.eval() 
        return self 

#Функция перевода режим обучения   
    def train(self, mode=True):        
        self.model.train(mode)
        return self
    
#Сохранение модели и скейлеров
#filepath - путь
    
    def save(self, filepath):
        
        if self.model is None:
            raise ValueError("Нет обученной модели для сохранения.")
        
        #сохранение весов модели
        torch.save(self.model.state_dict(), f'{filepath}_model_weights.pth')
        
        #сохранение скейлеров и параметров
        save_dict = {
            'scaler_X': self.scaler_X,
            'scaler_y': self.scaler_y,
            'input_size': self.input_size,
            'neural_layer_1': self.neural_layer_1,
            'neural_layer_2': self.neural_layer_2,
            'dropout1': self.dropout1,
            'dropout2': self.dropout2
        }
        joblib.dump(save_dict, f'{filepath}_model_config.joblib')
        
        print(f"Модель сохранена: {filepath}_model_weights.pth и {filepath}_model_config.joblib")
        
#Загрузка модели и скейлеров
#filepath - путь

    def load(self, filepath):
        
        #загрузка конфигураци
        config = joblib.load(f'{filepath}_model_config.joblib')
        
        #восстановление параметров
        self.scaler_X = config['scaler_X']
        self.scaler_y = config['scaler_y']
        self.input_size = config['input_size']
        self.neural_layer_1 = config['neural_layer_1']
        self.neural_layer_2 = config['neural_layer_2']
        self.dropout1 = config['dropout1']
        self.dropout2 = config['dropout2']
        
        #создание и загрузка модели
        self.model = self.KessonNN(
            self.input_size, 
            self.neural_layer_1, 
            self.neural_layer_2, 
            self.dropout1, 
            self.dropout2
        )
        self.model.load_state_dict(torch.load(f'{filepath}_model_weights.pth'))
        self.model.eval()
        
        print(f"Модель загружена: {filepath}")
        return self    
'''
        st.code(code_1, language="python")


with tab2:
    st.subheader("Обучение метамодели")
    code_2 = '''
    #обучение сети
    model.fit(X_train, y_train) 

    #вывод результатов обучения
    Epoch [20/200], Loss: 0.024781
    Epoch [40/200], Loss: 0.018015
    Epoch [60/200], Loss: 0.015669
    Epoch [80/200], Loss: 0.014050
    Epoch [100/200], Loss: 0.012718
    Epoch [120/200], Loss: 0.012753
    Epoch [140/200], Loss: 0.013055
    Epoch [160/200], Loss: 0.012262
    Epoch [180/200], Loss: 0.012284
    Epoch [200/200], Loss: 0.012037
        
        
        '''
    st.code(code_2, language="python")

    st.markdown(""" Сеть обучена за 200 эпох с размером батча 64:
    * начальный уровень ошибки: 0.0248
    * финальный уровень ошибки: 0.012 
    * снижение ошибки: ≈51.6% за 200 эпох
    * функция ошибки плавно уменьшается без резких скачков
    * начиная с эпохи 100, значение функции потерь стабилизируется в диапазоне 0.0120-0.0130
    """)
    
with tab3:    
    st.subheader("Проверка на тестовых данных")

    code_3 = '''
    #предсказания на тестовых данных
    predictions_test = model.predict(X_test, predictions_normalizer=True)
                
    #оценка предсказаний на тестовых данных
    score_valid = model.score(
        pred=predictions_test,
        y=y_test) 
        
    #вывод результатов
    MSE = 0.0020
    R2 масса = 0.9995
    R2 жесткость = 0.9965
        
        '''
    
    st.code(code_3, language="python")
    st.markdown(""" Результаты тестирования модели на тестовой выборке:
    * среднеквадратическая ошибка: *MSE* = 0.002 
    * коэффициент детерминации массы: *R²* = 0.9995
    * коэффициент детерминации жесткости: *R²* = 0.9965    
    """)
    st.subheader("Распределение остатков")
    st.markdown("""
    Для корректной оценки метамодели рассматривается распределение остатков.
    """)
    y_test = pd.read_csv("data/y_test.csv")
    predictions_test = pd.read_csv("data/predictions_test.csv")

    #визуализация остатков предсказаний массы конструкции
    fig_m = vis.charts_residuals_plotly(
        y_test['m'],
        predictions_test['predictions_m'],
        title='Масса, (т)'
    )

    st.plotly_chart(fig_m)     
    st.markdown("""По графику **Сравнение предсказаний с истинными значениями** наблюдается высокое совпадение предсказаний модели с реальными данными.  

Метамодель объясняет 99,95% дисперсии **целевой** переменной. Это прямое следствие наличия линейной зависимости **целевого** признака (массы) от **входящих** признаков (чем больше толщины силовых элементов, тем больше масса конструкции)""")

    #визуализация остатков предсказаний массы конструкции
    fig_stiffness = vis.charts_residuals_plotly(
        y_test['stiffness'],
        predictions_test['predictions_stiffness'],
        title='Жесткость, (Нмм²/рад)'
    )
    st.plotly_chart(fig_stiffness)

    st.markdown("""Анализ графика **Сравнение предсказаний с истинными значениями** подтверждает высокую точность модели: предсказанные значения демонстрируют тесную корреляцию с истинными данными.

Метамодель объясняет 99,65% дисперсии **целевой** переменной.
""")
    

    







    
    