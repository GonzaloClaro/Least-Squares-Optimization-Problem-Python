# -*- coding: utf-8 -*-
#Parte a.1
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

#definimos manualmente la matriz S de r(x) para x vector de R4
def S(x):
    #S es una matriz de 4x4, creamos una con 0s
    s=np.zeros((4,4,), dtype=int)
    r=R(x)

    #relleno las componentes de S que no son 0, implementando las sumatorias en forma de ciclo for
    for i  in range(0,N):
        aux1 = -r[i]*np.cos(x[1]*tdata[i]+x[2])*tdata[i]
        s[0,1] = s[0,1] + aux1
        s[1,0] = s[1,0] + aux1

        aux2 = -r[i]*np.cos(x[1]*tdata[i]+x[2])
        s[0,2] = s[0,2] + aux2
        s[2,0] = s[2,0] + aux2

        aux3 = r[i]*x[0]*tdata[i]*np.sin(x[1]*tdata[i]+x[2])
        s[1,2] = s[1,2]+aux3
        s[2,1] = s[1,2]+aux3

        s[1,1] = r[i]*x[0]*(tdata[i]**2)*np.sin(x[1]*tdata[i]+x[2])
        s[2,2] = r[i]*a*np.sin(x[1]*tdata[i]+x[2])

    return s


#Creamos la funcion recursiva para el metodo de Newton-Raphson
#recibe un valor objetivo x, un valor inicial x0 y un contador evaluado en 0 para las iteraciones
def newton(x0,x,contador=1):
    #definimos nuestros valores a utilizar, como son r(x), J(x) y S(x)
    r=R(x0)
    J=Jacobian(x0)
    s=S(x0)
    #np.dot multiplica matricialmente
    A=np.dot(np.transpose(J),J)
    AA=A+s
    #np.linalg.inv invierte la matriz
    B=np.linalg.inv(AA)
    C=np.dot(B,np.transpose(J))
    D=np.dot(C,r)

    #definimos un error para la diferencia entre componentes esperados y obtenidos mediante el metodo
    error = 10**-1
    
    #planteamos la recursion
    xx = x0 - D

    #si las componentes entre el x objetivo y el ajustado son pequeñas (del orden de error) se detiene el programa
    if ((math.sqrt((xx[0]-x[0])**2)<error)and(math.sqrt((xx[1]-x[1])**2)<error)and(math.sqrt((xx[2]-x[2])**2)<error)and(math.sqrt((xx[3]-x[3])**2)<error)):
        #entregamos algo de informacion
        print("Mediante Newton el x encontrado es:")
        print(xx)
        print("Con un total de iteraciones de:")
        print(contador)
        return xx
    
    #avisa que esta iterando si lleva una cierta cantidad de iteraciones ya hechas
    if contador==26:
        print("El programa esta iterando, por favor espere...")

    #si las iteraciones son muchas tambien se detiene
    if contador>100:
        print("Se realizan mas de 100 iteraciones, probar con otro valor")
        return 0
    #buscamos recursivamente ahora para el vector xx
    else:
        #manejamos un contador
        contador=contador+1
        return newton(xx,x,contador)


#definimos la funcion que va a graficar el ajuste obtenido para x0=1.001*x
def grafico_a1():
    x0 = np.array([a*1.001, omega*1.001, phi*1.001, b*1.001])
    datos_a1 = np.zeros(N)
    x_optimo_parte_a1 = newton(x0,x)
    for i in range(0,N):
        datos_a1[i] = Mo(x_optimo_parte_a1,tdata[i])
    plt.plot(tdata, datos_a1, color='red', marker='.', linestyle='', markersize=1)
    plt.plot(tdata, ydata, color='blue', marker='.', linestyle='', markersize=1)
    plt.xlabel("Tiempo")
    plt.ylabel("Temperatura")
    plt.title("Metodo Newton-Raphson")
    return plt.show()

#Ahora realizaremos las impresiones de los valores encontrados y descritos en el pdf de la tarea

print("-------------------------------------------------------------------------")
print("Tarea de Optimizacion parte A.1")
print("Gonzalo Claro")
print("Rut 19390187-5")
print("-------------------------------------------------------------------------")
print("")

print('El valor objetivo x, en forma de vector [a,omega,phi,b] es:')
print (x)


print("")
print("-------------------------------------------------------------------------")
print("")

#En esta seccion del codigo se implementa el metodo de Newton-Raphson de la parte a.1 para distintos valores de x0

print("Tras utilizar el metodo de Newton-Raphson para ajustar distintos valores de x0, los valores de x* optimo obtenidos en forma de vector [a, omega, phi, b] son:")

print("")

#utilizamos valores iniciales iguales a los que buscamos, es decir, x0=x
print("Usando x0=x=")
xigual = np.array([a,omega,phi,b])
print(xigual)
newton(xigual,x)

print("")

#utilizamos valores iniciales muy cercanos al valor x: x0=1.001*x
print("Usando valores muy cercanos: x0=1.001*x=")
x0 = np.array([a*1.001, omega*1.001, phi*1.001, b*1.001])
print(x0)
newton(x0,x)

print("")

#utilizamos valores iniciales un poco alejados al valor x: x0=1.1*x
print("Usando valores un poco alejados: x0=x*1.1=")
x00 = np.array([a*1.1,omega*1.1,phi*1.1,b*1.1])
print(x00)
newton(x00,x)

print("")

#utilizamos valores iniciales mas alejados al valor x: x0=7*x
print("Usando valores bastante alejados: x0=7*x=")
xlejano = np.array([a*7,omega*7,phi*7,b*7])
print(xlejano)
newton(xlejano,x)


print("")
print("-------------------------------------------------------------------------")
print("")

#En esta parte del codigo se itera "manualmente" mediante ciclos while segun lo comentado en el pdf.

print ('El valor de x*, en forma de vector [a, omega, phi, b] tras iterar mediante el algoritmo de Newton-Raphson mediante ciclos while es:')

print("")

#definimos el vector de condiciones iniciales a utilizar
xlejano = np.array([a*7,omega*7,phi*7,b*7])

#iteramos 4 veces el algoritmo de Newton para x0=7*x
print("Iterando 4 veces para x0 = 7*x es:")
i=0
x4=xlejano
while(i<4):
    r=R(x4)
    J=Jacobian(x4)
    s=S(x4)
    A=np.dot(np.transpose(J),J)
    AA=A+s
    B=np.linalg.inv(AA)
    C=np.dot(B,np.transpose(J))
    D=np.dot(C,r)
    xx = x4 - D
    x4=xx
    i=i+1
print(x4)

#iteramos 50 veces el algoritmo de Newton para x0=7*x
print("Iterando 50 veces para x0 = 7*x es:")
i=0
x50=xlejano
while(i<50):
    r=R(x50)
    J=Jacobian(x50)
    s=S(x50)
    A=np.dot(np.transpose(J),J)
    AA=A+s
    B=np.linalg.inv(AA)
    C=np.dot(B,np.transpose(J))
    D=np.dot(C,r)
    xx = x50 - D
    x50=xx
    i=i+1
print(x50)

print("")

#definimos el vector de condiciones iniciales a utilizar
x0 = np.array([a*1.001, omega*1.001, phi*1.001, b*1.001])

#iteramos 1 vez el algoritmo de Newton para x0=1.001*x
print("Iterando 1 vez para x0 = x*1.001 es:")
i=0
x1=x0
while(i<1):
    r=R(x1)
    J=Jacobian(x1)
    s=S(x1)
    A=np.dot(np.transpose(J),J)
    AA=A+s
    B=np.linalg.inv(AA)
    C=np.dot(B,np.transpose(J))
    D=np.dot(C,r)
    xx = x1 - D
    x1=xx
    i=i+1
print(x1)

#iteramos 5 veces el algoritmo de Newton para x0=1.001*x
print("Iterando 5 veces para x0 = x*1.001:")  
i=0
x5=x0
while(i<5):
    r=R(x5)
    J=Jacobian(x5)
    s=S(x5)
    A=np.dot(np.transpose(J),J)
    AA=A+s
    B=np.linalg.inv(AA)
    C=np.dot(B,np.transpose(J))
    D=np.dot(C,r)
    xx = x5 - D
    x5=xx
    i=i+1
print(x5)

#iteramos 15 veces el algoritmo de Newton para x0=1.001*x
print("Iterando 15 veces para x0 = x*1.001:")
x15=x0
i=0
while(i<15):
    r=R(x15)
    J=Jacobian(x15)
    s=S(x15)
    A=np.dot(np.transpose(J),J)
    AA=A+s
    B=np.linalg.inv(AA)
    C=np.dot(B,np.transpose(J))
    D=np.dot(C,r)
    xx = x15 - D
    x15=xx
    i=i+1
print (x15)

print("")
print("-------------------------------------------------------------------------")
print("")

#Al final graficamos, ya que el plot detiene la ejecucion del codigo
print("Finalmente se grafica el ajuste mediante Newton-Raphson para el vector de condiciones iniciales x0=1.001*x:")
print("")

grafico_a1()
