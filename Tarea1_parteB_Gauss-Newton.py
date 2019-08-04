# -*- coding: utf-8 -*-
#Parte b
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

#Definimos manualmente como es el Jacobiano de r(x) con x vector de R4
def Jacobian(x):
    #va a ser una matriz de 750x4
    Jacobiano = np.zeros((N,4))
    for i in range(0,N):
    #rellenamos su primera columna 
        Jacobiano[i,0] = -np.sin(x[1]*tdata[i]+x[2])
    #rellenamos su segunda columna
        Jacobiano[i,1] = -x[0]*np.cos(x[1]*tdata[i]+x[2])*tdata[i]
    #rellenamos su tercera columna
        Jacobiano[i,2] = -x[0]*np.cos(x[1]*tdata[i]+x[2])
    #rellenamos su cuarta columna
        Jacobiano[i,3] = -tdata[i]
    return Jacobiano

#Creamos la funcion recursiva para el metodo de Gauss-Newton
#recibe un valor objetivo x, un valor inicial x0 y un contador evaluado en 0 para las iteraciones
def newtonB(x0,x,contador=1):
    #definimos nuestros valores a utilizar, como son r(x), J(x)
    r=R(x0)
    J=Jacobian(x0)
    #np.dot multiplica matricialmente
    A=np.dot(np.transpose(J),J)
    #np.linalg.inv invierte la matriz
    B=np.linalg.inv(A)
    C=np.dot(B,np.transpose(J))
    D=np.dot(C,r)
    
    #definimos un error para la diferencia entre componentes esperados y obtenidos mediante el metodo
    error = 10**-1
    
    
    #planteamos la recursion
    xx = x0 - D

    #si las componentes entre el x objetivo y el ajustado son pequeñas (del orden de error) se detiene el programa
    if ((math.sqrt((xx[0]-x[0])**2)<error)and(math.sqrt((xx[1]-x[1])**2)<error)and(math.sqrt((xx[2]-x[2])**2)<error)and(math.sqrt((xx[3]-x[3])**2)<error)):
        #aportamos algo de informacion, aunque se pueden eliminar todos estos print para hacer el codigo mas limpio
        print("Mediante Newton parte B el x encontrado es:")
        print(xx)
        print("Con un total de iteraciones de:")
        print(contador)
        return xx

    #si las iteraciones son muchas tambien se detiene
    if contador>100:
        a="Se realizan mas de 100 iteraciones, probar con otro valor"
        print (a)
        return 0
    
    #buscamos recursivamente ahora para el vector xx
    else:
        #manejamos un contador
        contador=contador+1
        return newtonB(xx,x,contador)

#definimos la funcion que va a graficar el ajuste obtenido para x0=1.001*x
def grafico_b():
    x0 = np.array([a*1.001, omega*1.001, phi*1.001, b*1.001])
    datos_b = np.zeros(N)
    x_optimo_parte_b = newtonB(x0,x)
    for i in range(0,N):
        datos_b[i] = Mo(x_optimo_parte_b,tdata[i])
    plt.plot(tdata, datos_b, color='red', marker='.', linestyle='', markersize=1)
    plt.plot(tdata, ydata, color='blue', marker='.', linestyle='', markersize=1)
    plt.xlabel("Tiempo")
    plt.ylabel("Temperatura")
    plt.title("Metodo Gauss-Newton")
    return plt.show()

#Ahora realizaremos las impresiones de los valores encontrados y descritos en el pdf de la tarea

print("-------------------------------------------------------------------------")
print("Tarea de Optimizacion parte B")
print("Gonzalo Claro")
print("Rut 19390187-5")
print("-------------------------------------------------------------------------")
print("")

print('El valor objetivo x, en forma de vector [a,omega,phi,b] es:')
print (x)

print("")
print("-------------------------------------------------------------------------")
print("")

#En esta seccion del codigo se implementa el metodo de Gauss-Newton de la parte a.1 para distintos valores de x0

print("Tras utilizar el metodo de Gauss-Newton para ajustar distintos valores de x0, los valores de x* optimo obtenidos en forma de vector [a, omega, phi, b] son:")

print("")

#utilizamos valores iniciales iguales a los que buscamos, es decir, x0=x
print("Usando x0=x=")
xb2=np.array([a,omega,phi,b])
print(xb2)
newtonB(xb2,x)

print("")

#utilizamos valores iniciales muy cercanos al valor x: x0=1.001*x
print("Usando x0=1.001*x=")
xb=np.array([a*1.001,omega*1.001,phi*1.001,b*1.001])
print(xb)
newtonB(xb,x)

print("")

#utilizamos valores iniciales un poco alejados al valor x: x0=1.1*x
print("Usando x0=x*1.1=")
xb1=np.array([a*1.1,omega*1.1,phi*1.1,b*1.1])
print(xb1)
newtonB(xb1,x)

print("")

#utilizamos valores iniciales mas alejados al valor x: x0=7*x
print("Usando x0=7*x=")
xb3=np.array([a*7,omega*7,phi*7,b*7])
print(xb3)
newtonB(xb3,x)


print("")
print("-------------------------------------------------------------------------")
print("")

#En esta parte del codigo se itera "manualmente" mediante ciclos while segun lo comentado en el pdf.

print ('El valor de x*, en forma de vector [a, omega, phi, b] tras iterar mediante el algoritmo de Newton-Raphson mediante ciclos while es:')

print("")

#definimos el vector de condiciones iniciales a utilizar
x00_b = np.array([a*1.1, omega*1.1, phi*1.1, b*1.1])

#iteramos 4 veces el algoritmo de Newton para x0=1.1*x
print("Iterando 4 veces para x0 = 1.1*x:")
i=0
x4=x00_b
while(i<4):
    r=R(x4)
    J=Jacobian(x4)
    A=np.dot(np.transpose(J),J)
    B=np.linalg.inv(A)
    C=np.dot(B,np.transpose(J))
    D=np.dot(C,r)    
    xx = x4 - D
    x4=xx
    i=i+1
print(x4)

print("")

#definimos el vector de condiciones iniciales a utilizar 
xlejano_b = np.array([a*7,omega*7,phi*7,b*7])

#iteramos 4 veces el algoritmo de Newton para x0=7*x
print("Iterando 4 veces para x0 = 7*x:")
i=0
x4=xlejano_b
while(i<4):
    r=R(x4)
    J=Jacobian(x4)
    A=np.dot(np.transpose(J),J)
    B=np.linalg.inv(A)
    C=np.dot(B,np.transpose(J))
    D=np.dot(C,r)    
    xx = x4 - D
    x4=xx
    i=i+1
print(x4)

#iteramos 50 veces el algoritmo de Newton para x0=7*x
print("Iterando 50 veces para x0 = 7*x:")
i=0
x50=xlejano_b
while(i<50):
    r=R(x50)
    J=Jacobian(x50)
    A=np.dot(np.transpose(J),J)
    B=np.linalg.inv(A)
    C=np.dot(B,np.transpose(J))
    D=np.dot(C,r)    
    xx = x50 - D
    x50=xx
    i=i+1
print(x50)

print("")

#definimos el vector de condiciones iniciales a utilizar en las iteraciones
x0_b = np.array([a*1.001, omega*1.001, phi*1.001, b*1.001])

#iteramos 1 vez el algoritmo de Newton para x0=1.001*x
print("Iterando 1 vez para x0 = 1.001*x:")
i=0
x1=x0_b
while(i<1):
    r=R(x1)
    J=Jacobian(x1)
    A=np.dot(np.transpose(J),J)
    B=np.linalg.inv(A)
    C=np.dot(B,np.transpose(J))
    D=np.dot(C,r)    
    xx = x1 - D
    x1=xx
    i=i+1
print(x1)

#iteramos 5 veces el algoritmo de Newton para x0=1.001*x
print("Iterando 5 veces para x0 = 1.001*x:")
i=0
x5=x0_b
while(i<5):
    r=R(x5)
    J=Jacobian(x5)
    A=np.dot(np.transpose(J),J)
    B=np.linalg.inv(A)
    C=np.dot(B,np.transpose(J))
    D=np.dot(C,r)    
    xx = x5 - D
    x5=xx
    i=i+1
print(x5)

#iteramos 15 veces el algoritmo de Newton para x0=1.001*x
print("Iterando 15 veces para x0 = 1.001*x:")
i=0
x15=x0_b
while(i<15):
    r=R(x15)
    J=Jacobian(x15)
    A=np.dot(np.transpose(J),J)
    B=np.linalg.inv(A)
    C=np.dot(B,np.transpose(J))
    D=np.dot(C,r)    
    xx = x15 - D
    x15=xx
    i=i+1
print(x15)

print("")
print("-------------------------------------------------------------------------")
print("")

#Al final graficamos, ya que el plot detiene la ejecucion del codigo
print("Finalmente se grafica el ajuste mediante Gauss-Newton para el vector de condiciones iniciales x0=1.001*x:")
print("")

grafico_b()


