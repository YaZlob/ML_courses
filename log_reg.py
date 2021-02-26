from matplotlib import pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split

X, y = datasets.make_blobs(n_samples=10000, n_features=2, centers=2, random_state=42)
y = y[:, np.newaxis]
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

#plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train[:, 0])
#    plt.show()

class LogisticRegressionGD:
    '''
    A simple logistic regression for binary classification with gradient descent
    '''

    def __init__(self):
        pass

    def __extend_X(self, X):
        """
            Данный метод должен возвращать следующую матрицу:
            X_ext = [1, X], где 1 - единичный вектор
            это необходимо для того, чтобы было удобнее производить
            вычисления, т.е., вместо того, чтобы считать X@W + b
            можно было считать X_ext@W_ext
        """
        return np.hstack((np.zeros(shape=(X.shape[0],1)),X))

    def init_weights(self, input_size, output_size):
        """
            Инициализирует параметры модели
            W - матрица размерности (input_size, output_size)
            инициализируется рандомными числами из
            нормального распределения со средним 0 и стандартным отклонением 0.01
        """
        np.random.seed(42)
        self.W = np.random.normal(loc=0, scale=0.01, size=(input_size, output_size))  # YOUR_CODE

    def get_loss(self, p, y):
        """
            Данный метод вычисляет логистическую функцию потерь
            @param p: Вероятности принадлежности к классу 1
            @param y: Истинные метки
        """
        # YOUR_CODE
        # np.sum(y*np.log(p)+(1-y)*np.log(1-p))/y.shape[0]
        return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))

    def get_prob(self, X):
        """
            Данный метод вычисляет P(y=1|X,W)
            Возможно, будет удобнее реализовать дополнительный
            метод для вычисления сигмоиды
        """
        sigm = lambda x: 1 / (1 + np.exp(x))
        if X.shape[1] != self.W.shape[0]:
            X = self.__extend_X(X)
        # YOUR_CODE
        return sigm(-X@self.W)

    def get_acc(self, p, y, threshold=0.5):
        """
            Данный метод вычисляет accuracy:
            acc = \frac{1}{len(y)}\sum_{i=1}^{len(y)}{I[y_i == (p_i >= threshold)]}
        """
        # YOUR_CODE
        return (1 / y.shape[0]) * np.sum(y == (p >= threshold))

    def fit(self, X, y, num_epochs=100, lr=0.001):

        X = self.__extend_X(X)
        self.init_weights(X.shape[1], y.shape[1])
        print(self.W.shape)
        accs = []
        losses = []
        for _ in range(num_epochs):
            p = self.get_prob(X)

            W_grad = X.T @ (p - y) / y.shape[0]
            self.W -= lr * W_grad  # YOUR_CODE

            # необходимо для стабильности вычислений под логарифмом
            p = np.clip(p, 1e-10, 1 - 1e-10)

            log_loss = self.get_loss(p, y)
            losses.append(log_loss)
            acc = self.get_acc(p, y)
            accs.append(acc)

        return accs, losses

model = LogisticRegressionGD()
accs, losses = model.fit(X_train, y_train)
figure = plt.figure()
plt1 = figure.add_subplot(10,1,1)
plt2 = figure.add_subplot(2,1,2)

plt1.plot(accs)
plt1.set(title = "Точность")

plt2.plot(losses)
plt2.set(title = "Потери")

plt.show()