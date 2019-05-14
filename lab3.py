import matplotlib.pyplot as plt 
import pandas as pd 
import scipy as sc
import math

def gradien_descent(theta0=0,theta1=0,X,Y,f_x,alpha=0.1):
    for i in range(len(X)):
        theta0 = theta0 - alpha*1/m*sum(f_x(X[i]-Y[i]))
        theta1 = theta1 - alpha*1/m*sum((f_x(X[i]-Y[i])))*X[i]
        return (theta0, theta1)
    
data = pd.read_csv('ex1data1.csv', header=None) 
x, y= data[0], data[1] 
fy = lambda x: 1.5*x - 10
x1 = sc.linspace(min(x),max(x),10)
y1 = list(map(fy,x1))
fig = plt.figure()
pll = plt.subplot(111)
pll.plot(x,y,'b*')
pll.plot(x1,y1,'g-')
plt.savefig("test_plot_dot.png") 
plt.show()
print(sq_error(x,y,fy))
def sq_error(X,Y,f_x=None):
    squared_error = [];
    for i in range(len(X)):
        squared_error.append((f_x(X[i])-Y[i])**2)
    return sum(squared_error)   
