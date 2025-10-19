import pandas as pd
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler



delta_lista = [0.8, 0.9, 1.0, 1.2, 1.5, 1.6, 1.8, 1.9, 2.0, 2.5]



#обертка оптимизатора
class optimizer_kesson_model:
    
#Инициализация конвейера
    def __init__(
        self,
        model,  #обученная сеть
        lr=0.01,  #скорость обучения оптимизатора
        num_iterations=1000,  #количество итераций оптимизации
        clamp_min=-2.5,  #минимальное значение для зажимания параметров
        delta_technol=delta_lista  # технологические ограничения толщин
    ):
        
        self.model = model
        self.lr = lr
        self.num_iterations = num_iterations
        self.clamp_min = clamp_min        
        self.delta_technol = np.array(delta_technol)


#вычисление функции потерь
#x - входные данные 
#M_target_scaled - заданная стандартизированная масса 
    
    def optim_function(self, x, M_target_scaled):
        if not x.requires_grad:
                   x = x.requires_grad_(True)
        
        #предсказания
        predictions = self.model.predict_with_grad(x)
        
        #масса и жесткость - стандартизированные
        mass_pred_scaled = predictions[0, 0]
        stiffness_pred_scaled = predictions[0, 1]
    
        #штраф за отклонение от заданной массы
        penalty = (mass_pred_scaled - M_target_scaled) ** 2
        
        #функция потерь оптимизатора
        total_loss = -stiffness_pred_scaled + penalty

        results_optim_function = {
            'total_loss' : total_loss,
            'stiffness_pred_scaled' : stiffness_pred_scaled,
            'mass_pred_scaled' : mass_pred_scaled
        }
    
        return results_optim_function

#округление технологическое в минимальную сторону
#value - значение, найденное оптимизатором
    
    def round_to_allowed(self, value):
        idx = np.argmin(np.abs(self.delta_technol - value))
        return self.delta_technol[idx]

#прочностная оптимизация
#x - входной вектор в исходном масштабе (вокруг которого ищется оптимальное значние)
#M_target_scaled - заданная масса в стандартизированном масштабе
    
    def stress_optimization(self, x, M_target_scaled):
        
        #преобразование и нормализация входа
        x_original = np.array(x).reshape(1, -1)
        x_scaled = self.model.scaler_X.transform(x_original)
        
        #создание тензора с включенным градиентом
        x_scaled_tensor = torch.tensor(
            x_scaled,
            dtype=torch.float32,
            requires_grad=True
        )
        
        #перевод сети в режим оценки и заморозка весов
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

        #оптимизатор только для входных параметров
        optimizer = torch.optim.Adam([x_scaled_tensor], lr=self.lr)

        
        for i in range(1, self.num_iterations + 1):
            
            optimizer.zero_grad()

            #вычисление функции потерь
            results_optim_function = self.optim_function(
                x_scaled_tensor, M_target_scaled)
            
            total_loss = results_optim_function['total_loss']
            stiffness_pred_scaled = results_optim_function['stiffness_pred_scaled']
            mass_pred_scaled = results_optim_function['mass_pred_scaled']

            
            #обратное распространение и шаг оптимизатора
            total_loss.backward()
            optimizer.step()

            #корректировка значений (зажимание)
            with torch.no_grad():
                x_scaled_tensor.data = torch.clamp(
                    x_scaled_tensor.data,
                    self.clamp_min
                )


        return x_scaled_tensor

#корректировка подобранных значений ТЕХНОЛОГИЧЕСКАЯ
#x_scaled_tensor - параметры конструкции подобранные в ходе оптимизации

    def technological_optimization(self, x_scaled_tensor):

        with torch.no_grad():
            #преобразование обратно в исходный масштаб
            best_x_scaled = x_scaled_tensor.detach()
            best_x_original = self.model.scaler_X.inverse_transform(best_x_scaled.numpy())[0]

            #предсказание для непрерывных параметров (прочностная)
            final_pred_continuous = self.model.predict(
                best_x_original.reshape(1, -1), 
                predictions_normalizer=True
            )
            final_mass_continuous = final_pred_continuous[0, 0]
            final_stiffness_continuous = final_pred_continuous[0, 1]

            #округление каждого параметра до ближайшего допустимого технологического значения
            best_x_discrete = []
            for x in best_x_original:        
                best_x_discrete.append(self.round_to_allowed(x))
            
            best_x_discrete = np.array(best_x_discrete)
            
            #предсказание для округленных параметров
            final_pred_discrete = self.model.predict(
                best_x_discrete.reshape(1, -1),
                predictions_normalizer=True            
            )
            
            final_mass_discrete = final_pred_discrete[0, 0]
            final_stiffness_discrete = final_pred_discrete[0, 1]
            
            results = {
                'continuous_params': list(best_x_original),
                'discrete_params': list(best_x_discrete),
                'continuous_mass': final_mass_continuous,
                'continuous_stiffness': final_stiffness_continuous,
                'discrete_mass': final_mass_discrete,
                'discrete_stiffness': final_stiffness_discrete
            }


        return results

#полный процесс оптимизации
#x_initial - начальные параметры в исходном масштабе
#target_mass_scaled - заданная масштабированная масса 
    
    def optimize(self, x_initial, target_mass_scaled):

        
        #прочностная оптимизация
        optimized_params_scaled = self.stress_optimization(
            x_initial, target_mass_scaled)
        
        #технологическая оптимизация
        results = self.technological_optimization(optimized_params_scaled)
  
        return results