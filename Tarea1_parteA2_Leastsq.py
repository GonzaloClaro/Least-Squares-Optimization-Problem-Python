# -*- coding: utf-8 -*-
#Parte a.2
#Tarea Optimizacion Parte 1
#Gonzalo Claro
#rut 19.390.187-5

### importamos las librerías que usaremos ###
import numpy as np
import matplotlib.pyplot as plt
import random
import scipy as sc
from scipy.optimize import leastsq
import math 

# iniciamos la semilla para generar valores aleatorios
# recuerde cambiar '1' por su RUT
# es importante fijar la semilla para que los datos aleatorios generados
# siempre sean los mismos (de acuerdo a su RUT) y por lo tanto, sean replicables.

random.seed(19340187) # aca cambie a su rut

# Agregue su número de lista

nLista=22

# definimos los valores de los parámetros del modelo original
# estos son los parámetros que debemos buscar con los métodos de la tarea
# las 4 lineas que vienen NO pueden ser modificadas
a = (1+nLista/85)*8.967 + 0.05*random.random()
omega = 3.1415/3.0 - 0.1*random.random()
phi = 3.1415/6.0 + + 0.1*random.random()
b = (1+nLista/85)*0.345 - + 0.05*random.random()

#definimos la función del modelo M que asocia tiempo con temperatura
def M(t):
    ydata = a*np.sin(t*omega+phi) + b*t
    return ydata 
    
#definimos el arreglo de tiempo considerado, usted puede cambiar el intervalo
# de tiempo como el número de puntos utilizado
tdata = np.linspace(0.0, 25.0, num=750)
    
N = len(tdata)
ydata = np.zeros(N) #creamos el arreglo que contendrá los "datos medidos"
# los datos medidos en este caso son una perturbación de los datos originales
# esta perturbación es aleatoria y depende de su RUT

for i in range(0,N):
    rand = random.random()
    ydata[i] = M(tdata[i]) - 3.8546*rand*(-1)**i
    
#plt.plot(tdata, ydata, color='blue', marker='.', linestyle='', markersize=1)
#plt.show()

#definimos nuevas variables iniciales que representan las componentes de nuestro vector X0
#X0 es cercano al vector objetivo que queremos llegar para que funcione el metodo
a0 = a*(1.001)
omega0 = omega*(1.001)
phi0 = phi*(1.001)
b0 = b*(1.001)

#creamos el vector de condiciones iniciales 
x0 = np.array([a0, omega0, phi0, b0])

#definimos el vector objetivo al que queremos llegar con las variables originales
x = np.array([a, omega, phi, b])

#El nuevo modelo M se definira en base a un vector x dado
#x sera un vector de 4 componentes con las variables iniciales
def Mo(x,t):
    Modelo = x[0]*np.sin(t*x[1]+x[2]) + x[3]*t
    return Modelo

#Con el nuevo modelo Mo y las nuevas componentes de x recalculamos el valor de r
def R(x):
    r = np.zeros(N)
    for i in range(0,N):
        r[i] = ydata[i]-Mo(x,tdata[i])
    return np.transpose(r)

#IMPLEMENTACION DE LA FUNCION LEASTSQ

#lo que buscamos mediante leastsq es minimizar los residuos, es decir, la diferencia entre los datos obtenidos y el modelo
#esta diferencia es entregada mediante la funcion R(x)

print("-------------------------------------------------------------------------")
print("Tarea de Optimizacion parte A.2")
print("Gonzalo Claro")
print("Rut 19390187-5")
print("-------------------------------------------------------------------------")
print("")

#leastsq(f,x) recibe una funcion f que representa la diferencia entre los datos y el modelo, y tambien un vector de condiciones iniciales x para comenzar desde ahi el ajuste

#Implementamos la funcion leastsq para x=x0
print("El valor del optimo obtenido x*, para x0=x, en forma de vector [a, omega, phi, b] utilizando leastsq es:")
xigual = np.array([a,omega,phi,b])
ajuste = leastsq(R,xigual)
print(ajuste[0])

print("")

print("El valor del optimo obtenido x*, para x0=1.001*x, en forma de vector [a.ajuste, omega.ajuste, phi.ajuste, b.ajuste] utilizando leastsq es:")
x0 = np.array([a*1.001, omega*1.001, phi*1.001, b*1.001])
ajuste = leastsq(R,x0)
print(ajuste[0])
print("")
print("El valor del optimo obtenido x*, para x0=1.1*x, en forma de vector [a, omega, phi, b] utilizando leastsq es:")
x00 = np.array([a*1.1,omega*1.1,phi*1.1,b*1.1])
ajuste = leastsq(R,x00)
print(ajuste[0])
print("")
print("El valor del optimo obtenido x*, para x0=7*x, en forma de vector [a, omega, phi, b] utilizando leastsq es:")
xlejano = np.array([a*7,omega*7,phi*7,b*7])
ajuste = leastsq(R,xlejano)
print(ajuste[0])

#definimos la funcion que va a graficar el ajuste obtenido para x0=1.001*x
def grafico_a2():
    x0 = np.array([a*1.001, omega*1.001, phi*1.001, b*1.001])
    datos_a2 = np.zeros(N)
    x_optimo_parte_a2 = leastsq(R,x0)
    for i in range(0,N):
        datos_a2[i] = Mo(x_optimo_parte_a2[0],tdata[i])
    plt.plot(tdata, datos_a2, color='red', marker='.', linestyle='', markersize=1)
    plt.plot(tdata, ydata, color='blue', marker='.', linestyle='', markersize=1)
    plt.xlabel("Tiempo")
    plt.ylabel("Temperatura")
    plt.title("Leastsq")
    return plt.show()

print("")
print("-------------------------------------------------------------------------")
print("")

#Al final graficamos, ya que el plot detiene la ejecucion del codigo
print("Finalmente se grafica el ajuste mediante leastsq para el vector de condiciones iniciales x0=1.001*x:")

print("")

grafico_a2()
