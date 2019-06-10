import matplotlib.pyplot as plt 
import pandas as pd 
import scipy as sc
import numpy as np
import math

# Загрузка исходных данных из файла ex1data1.csv

data = pd.read_csv('ex1data1.csv', header=None) 

x, y= data[0], data[1] 

# Построение линейной функции 

fy = lambda x: 1.19*x - 3.89

x1 = sc.linspace(min(x),max(x),10)
y1 = list(map(fy,x1))

fig = plt.figure()
pll = plt.subplot(111)

pll.plot(x,y,'b*')

pll.plot(x1,y1,'g-')

plt.show()

# Алгоритм градиентного спуска для линейной регрессии с одной переменной. Нахождение theta0 и theta1

def gradient_descent(X, Y, koef, n):
    l = len(x)
    theta0, theta1 = 0, 0
    for i in range(n):
        sum1 = 0
        for i in range(l):
            sum1 += theta0 + theta1 * x[i] - y[i]
        res1 = theta0 - koef * (1 / l) * sum1

        sum2 = 0
        for i in range(l):
            sum2 += (theta0 + theta1 * x[i] - y[i]) * x[i]
        res2 = theta1 - koef * (1 / l) * sum2

        theta0, theta1 = res1, res2

    return theta0, theta1

# Вычисление коэффициента среднеквадратического отклонения

def sq_error(X,Y,f_x=None):
    squared_error = [];
    for i in range(len(X)):
        squared_error.append((f_x(X[i])-Y[i])**2)
    return sum(squared_error)  

print(f"Ср. кв. ошибка составляет = {sq_error(x, y, fy)}")

# Метод polyfit (степень полинома = 1)
x1,y1 = [0,22.5],[0,25] 

numpy_x = np.array(x)
numpy_y = np.array(y)

numpy_t1, numpy_t0 = np.polyfit(numpy_x, numpy_y, 1)

num_y1 = [0, 0]
num_y1[0] = numpy_t0 + x1[0] * numpy_t1
num_y1[1] = numpy_t0 + x1[1] * numpy_t1
plt.plot(x1, num_y1, 'b')

print('Коэффициенты, полученные с помощью polyfit:', numpy_t0, numpy_t1)

# Градиентный спуск

x2,y2 = [1, 25],[0, 0]
t0, t1 = gradient_descent(x, y, 0.01, len(x))
y2[0] = t0 + x2[0] * t1
y2[1] = t0 + x2[1] * t1
plt.plot(x2, y2, 'r')

# Построение графиков
fig = plt.plot(x, y, 'g*') 
fig1 = plt.plot(x1, y1, 'y') 
plt.show()
