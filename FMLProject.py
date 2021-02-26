import torch
import pandas as pd
import numpy as np
from torch import functional as F
from torch import nn
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score

#torch.cuda.current_device()
# Создание нескольких блоков данных, чтобы
#не все передавать модели на обучение,
#так данные легче обработать
def batch_generator(X, y, batch_size):
    np.random.seed(42)
    #perm = np.random.permutation(len(X))
    split_num = len(X)//batch_size
    if len(X)%batch_size != 0 :
        split_num+=1
    y = y.reshape(-1, 1)
    volume = np.hstack((X,y))
    np.random.shuffle(volume)
    volume = np.array_split(volume,split_num)
    for i in range(len(volume)):
        yield volume[i][:,:-1],volume[i][:,-1]



# Тестирование batch_generatora
from inspect import isgeneratorfunction
assert isgeneratorfunction(batch_generator), "batch_generator должен быть генератором! В условии есть ссылка на доки"

X = np.array([
              [1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]
])
y = np.array([
              1, 2, 3
])

# Проверим shape первого батча
iterator = batch_generator(X, y, 2)
X_batch, y_batch = next(iterator)
assert X_batch.shape == (2, 3), y_batch.shape == (2,)
assert np.allclose(X_batch, X[:2]), np.allclose(y_batch, y[:2])

# Проверим shape последнего батча (их всего два)
X_batch, y_batch = next(iterator)
assert X_batch.shape == (1, 3), y_batch.shape == (1,)
assert np.allclose(X_batch, X[2:]), np.allclose(y_batch, y[2:])

# Проверим, что итерации закончились
iter_ended = False
try:
    next(iterator)
except StopIteration:
    iter_ended = True
assert iter_ended

# Еще раз проверим то, сколько батчей создает итератор
X = np.random.randint(0, 100, size=(1000, 100))
y = np.random.randint(-1, 1, size=(1000, 1))
num_iter = 0
for _ in batch_generator(X, y, 3):
    num_iter += 1

assert num_iter == (1000 // 3 + 1)



feature_columns = ['ra', 'dec', 'u', 'g', 'r', 'i', 'z', 'run', 'camcol', 'field']
target_column = 'class'

target_mapping = \
    {
    'GALAXY': 0,
    'STAR': 1,
    'QSO': 2
}

# Считываю данные из выбранного датасета
# полагаю ,что разделение идет по координатам, но не уверен
data = pd.read_csv('sky_data.csv')
#print(data[:5])
data['class'].value_counts()
#print(data.columns)
X = data[feature_columns]
y = data[target_column].replace(target_mapping).values
X = ((X-X.mean())/X.std(ddof=0)).to_numpy()

#снова проверка на корректность
#работы кода
assert type(X) == np.ndarray and type(y) == np.ndarray, 'Проверьте, что получившиеся массивы являются np.ndarray'
assert np.allclose(y[:5], [1,1,0,1,1])
assert X.shape == (10000, 10)
assert np.allclose(X.mean(axis=0), np.zeros(10)) and np.allclose(X.std(axis=0), np.ones(10)), 'Данные не отнормированы'

# Split train/test
#разделение данных на тренировочную часть
#и тестовую
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
# Превратим данные в тензоры, чтобы потом было удобнее

print(X_train[0])

X_train = torch.FloatTensor(X_train)
y_train = torch.LongTensor(y_train)
X_test = torch.FloatTensor(X_test)
y_test = torch.LongTensor(y_test)


#gen = batch_generator(X_train,y_train,500)
#i,j = next(gen)
#print(i.shape,j.shape)


# Создание модели
torch.manual_seed(42)
np.random.seed(42)

model = nn.Sequential\
(
        nn.Linear(10, 50),
        nn.ReLU(),
        nn.Dropout(p=0.5),
        nn.BatchNorm1d(50),


        nn.Linear(50, 100),
        nn.ReLU(),
        nn.Dropout(p=0.5),
        nn.BatchNorm1d(100),

        nn.Linear(100, 3)
)
"""
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

# обучение модели
def train(X_train, y_train, X_test, y_test, num_epoch):
    train_losses = []
    test_losses = []
    for i in range(num_epoch):
        epoch_train_losses = []
        for X_batch, y_batch in batch_generator(X_train, y_train, 500):
            # На лекции мы рассказывали, что дропаут работает по-разному во время обучения и реального предсказания
            # Чтобы это учесть нам нужно включать и выключать режим обучения, делается это командой ниже
            model.train(True)
            # Посчитаем предсказание и лосс
            # YOUR CODE
            y_pred = model.forward(torch.from_numpy(X_batch).float())
            loss = loss_fn(y_pred,torch.tensor(y_batch).long())

            # зануляем градиент
            # YOUR CODE
            optimizer.zero_grad()
            # backward
            # YOUR CODE
            loss.backward()
            # ОБНОВЛЯЕМ веса
            # YOUR CODE
            optimizer.step()
            # Запишем число (не тензор) в наши батчевые лоссы
            epoch_train_losses.append(loss.item())  # YOUR CODE)
        train_losses.append(np.mean(epoch_train_losses))

            # Теперь посчитаем лосс на тесте
        model.train(False)
        with torch.no_grad():
            # Сюда опять же надо положить именно число равное лоссу на всем тест датасете
            test_losses.append(np.mean(loss_fn(model(X_test),y_test).item()))  # YOUR CODE)

    return train_losses, test_losses

def check_loss_decreased():
    print("На графике сверху, точно есть сходимость? Точно-точно? [Да/Нет]")
    s = input()
    if s.lower() == 'да':
        print("Хорошо!")
    else:
        raise RuntimeError("Можно уменьшить дропаут, уменьшить lr, поправить архитектуру, etc")


train_losses, test_losses = train(X_train,y_train,X_test,y_test,200) #обученная модель
# YOUR CODE) #Подберите количество эпох так, чтобы график loss сходился
# Вывод функции потерь
plt.plot(range(len(train_losses)), train_losses, label='train')
plt.plot(range(len(test_losses)), test_losses, label='test')
plt.legend()
#plt.show()
#
#check_loss_decreased()
assert train_losses[-1] < 0.3 and test_losses[-1] < 0.3

model.eval()
train_pred_labels = model.forward(X_train).max(1)[1]#YOUR CODE: use forward
test_pred_labels = model.forward(X_test).max(1)[1]#YOUR CODE: use forward

#расчет точности
train_acc = accuracy_score(y_train,train_pred_labels)
test_acc = accuracy_score(y_test,test_pred_labels)

assert train_acc > 0.9, "Если уж классифицировать звезды, которые уже видел, то не хуже, чем в 90% случаев"
assert test_acc > 0.9, "Новые звезды тоже надо классифицировать хотя бы в 90% случаев"

print("Train accuracy: {}\nTest accuracy: {}".format(train_acc, test_acc))
"""
"""
НЕПРАВИЛЬНЫЙ ПРИМЕР, КОТОРЫЙ ПЕРЕДЕЛАН НИЖЕ
torch.manual_seed(42)   
np.random.seed(42)
# WRONG ARCH
model = nn.Sequential(

    nn.Dropout(p=0.5),
    nn.Linear(6, 50),
    nn.ReLU(),
    nn.Dropout(p=0.5),
    nn.Linear(100, 200),
    nn.Softmax(),
    nn.Linear(200, 200),
    nn.ReLU(),
    nn.Dropout(p=0.5),
    nn.Linear(200, 3),
    nn.Dropout(p=0.5)
)


loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters[:-2], lr=1e-100)
"""
"""
#Здесь нужно было переделать нейросеть
#чтобы она заработала,закоментирована 
#неправильная архитектура
# RIGHT ARCH
#torch.manual_seed(42)
#np.random.seed(42)
model = nn.Sequential(

    nn.Linear(10, 50),
    nn.ReLU(),
    nn.Dropout(p=0.5),

    nn.Linear(50, 200),
    nn.ReLU(),
    nn.Dropout(p=0.5),

    nn.Linear(200, 200),
    nn.ReLU(),
    nn.Dropout(p=0.5),

    nn.Linear(200, 3)
)

# Тренировка модели с другой архитектурой, 
# те же тесты по точности
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),lr=1e-2)

train_losses, test_losses = train(X_train,y_train,X_test,y_test,200)

model.eval()
train_pred_labels = model.forward(X_train).max(1)[1]#YOUR CODE: use forward
test_pred_labels = model.forward(X_test).max(1)[1]#YOUR CODE: use forward

train_acc = accuracy_score(y_train,train_pred_labels)
test_acc = accuracy_score(y_test,test_pred_labels)

assert train_acc > 0.9, "Если уж классифицировать звезды, которые уже видел, то не хуже, чем в 90% случаев"
assert test_acc > 0.9, "Новые звезды тоже надо классифицировать хотя бы в 90% случаев"

print("Train accuracy: {}\nTest accuracy: {}".format(train_acc, test_acc))


torch.manual_seed(42)
np.random.seed(42)
"""
""" 
#Задание на определение оптимального 
#количества скрытых слоев в нейронке
#график выводт функцию потерь

Flag = True

min_loss = []

while Flag:
    head = nn.Sequential(
    nn.Linear(10, 100),
    nn.ReLU(),
    nn.Dropout(p=0.5))

    tail = nn.Sequential(nn.Linear(100,3))
    for i in range(1,21):
        # create blocks
        block = i*[ nn.Sequential(
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Dropout(p=0.5))]

        model = nn.Sequential(head,*block,tail)
        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
        train_losses, test_losses = train(X_train, y_train, X_test, y_test, 20)

        min_loss.append(min(test_losses))

        if i == 6:
            Flag = False
            break
        print(len(block))

plt.plot(range(len(min_loss)), min_loss, label='min_loss')
plt.legend()
plt.show()
# Дополнительные блоки (в моем случае начиная со второго), увеличивают Loss
#Я считаю, что потери увеличиваются из-за переобучения,
#которое возникает вследствие ненужного усложнения модели.
"""