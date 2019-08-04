# -*- coding: utf-8 -*-
#Parte c
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

print("-------------------------------------------------------------------------")
print("Tarea de Optimizacion parte C")
print("Gonzalo Claro")
print("Rut 19390187-5")
print("-------------------------------------------------------------------------")
print("")

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


#definimos la funcion que va a graficar el ajuste obtenido para x0=1.001*x mediante Newton-Raphson
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

#definimos la funcion que va a graficar el ajuste obtenido para x0=1.001*x mediante leastsq
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

#creamos un while para ejecutar un menu de opcines

while True:
    print("")
    print("-------------------------------------------------------------------------")
    print("")
    print("Hola, bienvenido a tu ajustador de datos. Escoge una opcion para graficar")
    print("")
    print("1) Realizar un ajuste mediante Newton-Raphson (parte a.1)")
    print("2) Realizar un ajuste mediante leastsq (parte a.2)")
    print("3) Realizar un ajuste mediante Gauss-Newton (parte b)")
    print("")
    a=input("¿Que opcion quieres graficar? escribe tu eleccion (1, 2 o 3): ")
    if a==1:
        print("")
        grafico_a1()
    elif a==2:
        print("")
        grafico_a2()
    elif a==3:
        print("")
        grafico_b()
    else:
        print("Has ingresado mal una opcion, vuelve a intentarlo!")
