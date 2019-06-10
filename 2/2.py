import matplotlib.pyplot as plt 
import pandas as pd 
import numpy as np
import math
import csv

# Считываем данные с файла web_traffic.tsv.

df = pd.read_csv('web_traffic.tsv', sep='\t', header=None)

X, Y = df[0], df[1]

x = list(X)
y = list(Y)

# Фильтрация точек

for i in range(len(y)):
    if math.isnan(y[i]):
        y[i] = 0
    else:
        y[i] = y[i]
        
# Визуализация точек

plt.plot(x, y, 'k*')

# Создание массивов

numpy_x = np.array(x)
numpy_y = np.array(y)

#  Подбор коэффициентов с помощью метода polyfit

th0, th1 = np.polyfit(numpy_x, numpy_y, 1)
th2, th3, th4 = np.polyfit(numpy_x, numpy_y, 2)
th5, th6, th7, th8 = np.polyfit(numpy_x, numpy_y, 3)
th9, th10, th11, th12, th13 = np.polyfit(numpy_x, numpy_y, 4)
th14, th15, th16, th17, th18, th19 = np.polyfit(numpy_x, numpy_y, 5)

f1 = lambda x: th0*x + th1
f2 = lambda x: th2*x**2 + th3*x + th4
f3 = lambda x: th5*x**3 + th6*x**2 + th7*x + th8
f4 = lambda x: th9*x**4 + th10*x**3 + th11*x**2 + th12*x + th13
f5 = lambda x: th14*x**5 + th15*x**4 + th16*x**3 + th17*x**2 + th18*x + th19

# Вычисление среднеквадратичной ошибки 

def sq_error(X,Y,f_x=None): 
    squared_error = []; 
    for i in range(len(X)): 
        squared_error.append((f_x(X[i])-Y[i])**2) 
    return sum(squared_error)

print(f"Ср. кв. ошибка при полиноме = 1  составляет : {sq_error(x, y, f1)}")
print(f"Ср. кв. ошибка при полиноме = 2  составляет : {sq_error(x, y, f2)}")
print(f"Ср. кв. ошибка при полиноме = 3  составляет : {sq_error(x, y, f3)}")
print(f"Ср. кв. ошибка при полиноме = 4  составляет : {sq_error(x, y, f4)}")
print(f"Ср. кв. ошибка при полиноме = 5  составляет : {sq_error(x, y, f5)}")

# Отображение функций при разных значениях полиномы

x1 = list(range(744, 751))

func1 = np.poly1d(np.polyfit(numpy_x, numpy_y, 1))
plt.plot(x1, func1(x1))

func2 = np.poly1d(np.polyfit(numpy_x, numpy_y, 2))
plt.plot(x1, func2(x1))

func3 = np.poly1d(np.polyfit(numpy_x, numpy_y, 3))
plt.plot(x1, func3(x1))

func4 = np.poly1d(np.polyfit(numpy_x, numpy_y, 4))
plt.plot(x1, func4(x1))

func5 = np.poly1d(np.polyfit(numpy_x, numpy_y, 5))
plt.plot(x1, func5(x1))

plt.show()
