import torch


def find_x_derivative(x, y):
    x = torch.tensor(x, dtype=float, requires_grad=True)
    out = torch.sin(torch.tan(x) * (x * x / y) + torch.log(torch.exp(-x * x + 3) + x * x * x * y)) * torch.tan(
        x * x * (torch.exp(x ** 9)))
    out.backward()
    return x.grad


# print(find_x_derivative(56,21))

def new_matrixA(matrix):
    _ = []
    for item in matrix:
        row_norm = torch.norm(item, keepdim=True).item()
        _.append(np.array(item.numpy() / row_norm))
    return _


def get_cos_sim_1(A, B):
    a = torch.tensor(A, dtype=float).split(1, dim=0)
    b = torch.tensor(B, dtype=torch.double)
    new_a = torch.tensor(new_matrixA(a)).reshape(4, 4)
    return new_a @ b


A = [[1, -47, 25, -3], [10, 17, -15, 22], [-3, -7, 26, 36], [12, -27, -42, 0]]

B_ = [[-50, -13, 1, 10, 1242],
      [21, 48, -13, -14, -20],
      [20, 15, 11, 43, 11],
      [11, 103, 147, 27, -8]]

B = [[-0.8498, -0.1127, 0.0068, 0.1865, 0.9998],
     [0.3569, 0.4161, -0.0878, -0.2611, -0.016],
     [0.3400, 0.1300, 0.0743, 0.8020, 0.0089],
     [0.1870, 0.8929, 0.9933, 0.5036, -0.0064]]
# Решение еще проще : перемножить матрицы , поделить на скалярное произведение норм
""""
return (A @ B) / (A.norm(dim=1, keepdim=True) @ B.norm(dim=0, keepdim=True))
"""
#print(torch.mean(get_cos_sim_1(A, B)))

# part 3

import numpy as np
from sklearn.datasets import make_regression

class LinearRegression:
    def get_loss(self, preds, y):
        """
            @param preds: предсказания модели
            @param y: истиные значения
            @return mse: значение MSE на переданных данных
        """
        # возьмите средний квадрат ошибки по всем выходным переменным
        # то есть сумму квадратов ошибки надо поделить на количество_элементов * количество_таргетов
        #return ((preds - y) ** 2).sum() / y.shape[0] numpy
        return torch.sum((preds-y)**2)/torch.numel(y)

    def init_weights(self, input_size, output_size):
        """
            Инициализирует параметры модели
            W - матрица размерности (input_size, output_size)
            инициализируется рандомными числами из
            uniform распределения (torch.rand())
            b - вектор размерности (1, output_size)
            инициализируется нулями
        """
        torch.manual_seed(0)  # необходимо для воспроизводимости результатов
        self.W = torch.rand(input_size, output_size, requires_grad=True)
        self.b = torch.zeros(1, output_size, requires_grad=True)

    def fit(self, X, y, num_epochs=1000, lr=0.001):
        """
            Обучение модели линейной регрессии методом градиентного спуска
            @param X: размерности (num_samples, input_shape)
            @param y: размерности (num_samples, output_shape)
            @param num_epochs: количество итераций градиентного спуска
            @param lr: шаг градиентного спуска
            @return metrics: вектор значений MSE на каждом шаге градиентного
            спуска.
        """
        self.init_weights(X.shape[1], y.shape[1])
        metrics = []
        for _ in range(num_epochs):
            preds = self.predict(X)
            # сделайте вычисления градиентов c помощью Pytorch и обновите веса
            # осторожнее, оберните вычитание градиента в
            #                 with torch.no_grad():
            #                     #some code
            # иначе во время прибавления градиента к переменной создастся очень много нод в дереве операций
            # и ваши модели в будущем будут падать от нехватки памяти
            # YOUR_CODE
            # YOUR_CODE
            # YOUR_CODE
            # YOUR_CODE
            #W_grad = 2*X.T@(preds-y)/X.shape[0]
            #b_grad =np.mean(2*(preds-y),axis = 0)
            loss = self.get_loss(preds,y)
            loss.backward()
            W_grad = self.W.grad
            b_grad = self.b.grad
            with torch.no_grad():
                self.W -= lr * W_grad
                self.b -= lr * b_grad
            metrics.append(self.get_loss(preds, y).data)
            self.W.grad.data.zero_()
            self.b.grad.data.zero_()
        return metrics

    def predict(self, X):
        """
            Думаю, тут все понятно. Сделайте свои предсказания :)
        """
        return X@self.W + self.b

X,Y = make_regression(n_targets=3, n_features=2, noise=10, random_state=42)
X = torch.tensor(X,dtype=torch.float)
Y = torch.tensor(Y,dtype=torch.float)
model = LinearRegression()
mse = model.fit(X,Y)
print(mse)