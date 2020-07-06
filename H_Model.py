# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 09:58:28 2019

@author: xialumi
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integer

N = 800     #Nombre de spires
mu = 4*np.pi* (10**(-7))        #perméabilité magnétique de l'air
#I1 = 5      #Intensité du courant dans les anti-Helmholtz
I2 = 0      #Intensité du courant dans les Helmholtz
a = 0.0625      #Rayon des spires    
e2 = 1.5*a      #Distance entre les bobines
alpha = 1       #Rapport entre les courants des bobines
suscep = 5.4*(10**(-4))  #Mass magnetic susceptibility of the magnetic beads
vol = (4/3)*np.pi*(((2.83/2)*(10**(-6)))**3)  #Volume of a magnetic bead


#Magnetic field created by Helmholtz
#component Bx
def Bx_H1(x, y, z):
    f = lambda theta : (N * mu * I2 * a/(4*np.pi)) *((z-(e2/2)) * np.cos(theta))/(((x - a*np.cos(theta))**2 + (y - a*np.sin(theta))**2 + (z-(e2/2))**2)**(3/2))
    res = integer.quad(f, -np.pi, np.pi)[0]  #Calcul de l'intégrale elliptique de 1ère espèce
    return(res)
def Bx_H2(x, y, z):
    f = lambda theta : (N * mu * I2 * a/(4*np.pi)) *((z+(e2/2)) * np.cos(theta))/(((x - a*np.cos(theta))**2 + (y - a*np.sin(theta))**2 + (z+(e2/2))**2)**(3/2))
    res = integer.quad(f, -np.pi, np.pi)[0]
    return(res)
def Bx_H(x, y, z):
    return(Bx_H1(x, y, z) + alpha*Bx_H2(x, y, z))

#component By
def By_H1(x, y, z):
    f = lambda theta : (N * mu * I2 * a/(4*np.pi)) *((z-(e2/2)) * np.sin(theta))/(((x - a*np.cos(theta))**2 + (y - a*np.sin(theta))**2 + (z-(e2/2))**2)**(3/2))
    res = integer.quad(f, -np.pi, np.pi)[0]  #Calcul de l'intégrale elliptique de 1ère espèce
    return(res)
def By_H2(x, y, z):
    f = lambda theta : (N * mu * I2 * a/(4*np.pi)) *((z+(e2/2)) * np.sin(theta))/(((x - a*np.cos(theta))**2 + (y - a*np.sin(theta))**2 + (z+(e2/2))**2)**(3/2))
    res = integer.quad(f, -np.pi, np.pi)[0]
    return(res)    
def By_H(x, y, z):
    return(By_H1(x, y, z) + alpha*By_H1(x, y, z))

#component Bz   
def Bz_H1(x, y, z):
    f = lambda theta : (N * mu * I2 * a/(4*np.pi)) *(a - x*np.cos(theta) - y*np.sin(theta))/(((x - a*np.cos(theta))**2 + (y - a*np.sin(theta))**2 + (z-(e2/2))**2)**(3/2))
    res = integer.quad(f, -np.pi, np.pi)[0]
    return(res)
def Bz_H2(x, y, z):
    f = lambda theta : (N * mu * I2 * a/(4*np.pi)) *(a - x*np.cos(theta) - y*np.sin(theta))/(((x - a*np.cos(theta))**2 + (y - a*np.sin(theta))**2 + (z+(e2/2))**2)**(3/2))
    res = integer.quad(f, -np.pi, np.pi)[0]
    return(res)
def Bz_H(x, y, z):
    return(Bz_H1(x, y, z) + alpha*Bz_H2(x, y, z))


#Gradient of magnetic field created by Helmholtz
#Gradient of Bx over x    
def dBx_dx_H1(x, y, z):
    f = lambda theta : -(N * mu * I2 * a/(4*np.pi)) * (3*(x-a*np.cos(theta))*(z-(e2/2))*np.cos(theta))/(((x-a*np.cos(theta))**2 + (y-a*np.sin(theta))**2 + (z-(e2/2))**2)**(5/2))
    res = integer.quad(f,-np.pi,np.pi)[0]
    return(res)
def dBx_dx_H2(x, y, z):
    f = lambda theta : -(N * mu * I2 * a/(4*np.pi)) * (3*(x-a*np.cos(theta))*(z+(e2/2))*np.cos(theta))/(((x-a*np.cos(theta))**2 + (y-a*np.sin(theta))**2 + (z+(e2/2))**2)**(5/2))
    res = integer.quad(f,-np.pi,np.pi)[0]
    return(res)
def dBx_dx_H(x,y,z):
    return(dBx_dx_H1(x,y,z) + alpha*dBx_dx_H2(x,y,z))

#Gradient of Bx over y 
def dBx_dy_H1(x,y,z):
    f = lambda theta : -(N * mu * I2 * a/(4*np.pi)) * (3*(y-a*np.sin(theta))*(z-(e2/2))*np.cos(theta))/(((x-a*np.cos(theta))**2 + (y-a*np.sin(theta))**2 + (z-(e2/2))**2)**(5/2))
    res = integer.quad(f,-np.pi,np.pi)[0]
    return(res)
def dBx_dy_H2(x,y,z):
    f = lambda theta : -(N * mu * I2 * a/(4*np.pi)) * (3*(y-a*np.sin(theta))*(z+(e2/2))*np.cos(theta))/(((x-a*np.cos(theta))**2 + (y-a*np.sin(theta))**2 + (z+(e2/2))**2)**(5/2))
    res = integer.quad(f,-np.pi,np.pi)[0]
    return(res)
def dBx_dy_H(x,y,z):
    return(dBx_dy_H1(x,y,z) + alpha*dBx_dy_H2(x,y,z))

#Gradient of Bx over xz
def dBx_dz_H1(x,y,z):
    f = lambda theta : (N * mu * I2 * a/(4*np.pi)) * ((np.cos(theta)*((x-a*np.cos(theta))**2 + (y-np.sin(theta))**2 + (z-(e2/2))**2)) - 3 * ((z-(e2/2))**2)*np.cos(theta))/(((x-a*np.cos(theta))**2 + (y-a*np.sin(theta))**2 + (z-(e2/2))**2)**(5/2))
    res = integer.quad(f,-np.pi,np.pi)[0]
    return(res)
def dBx_dz_H2(x,y,z):
    f = lambda theta : (N * mu * I2 * a/(4*np.pi)) * ((np.cos(theta)*((x-a*np.cos(theta))**2 + (y-np.sin(theta))**2 + (z+(e2/2))**2)) - 3 * ((z+(e2/2))**2)*np.cos(theta))/(((x-a*np.cos(theta))**2 + (y-a*np.sin(theta))**2 + (z+(e2/2))**2)**(5/2))
    res = integer.quad(f,-np.pi,np.pi)[0]
    return(res)
def dBx_dz_H(x,y,z):
    return(dBx_dz_H1(x,y,z) + alpha*dBx_dz_H2(x,y,z))

#Gradient of By over x 
def dBy_dx_H1(x,y,z):
    f = lambda theta : -(N * mu * I2 * a/(4*np.pi)) * (3*(x-a*np.cos(theta))*(z-(e2/2))*np.cos(theta))/(((x-a*np.cos(theta))**2 + (y-a*np.sin(theta))**2 + (z-(e2/2))**2)**(5/2))
    res = integer.quad(f,-np.pi,np.pi)[0]
    return(res)
def dBy_dx_H2(x,y,z):
    f = lambda theta : -(N * mu * I2 * a/(4*np.pi)) * (3*(x-a*np.cos(theta))*(z+(e2/2))*np.cos(theta))/(((x-a*np.cos(theta))**2 + (y-a*np.sin(theta))**2 + (z+(e2/2))**2)**(5/2))
    res = integer.quad(f,-np.pi,np.pi)[0]
    return(res)
def dBy_dx_H(x,y,z):
    return(dBy_dx_H1(x,y,z) + alpha*dBy_dx_H2(x,y,z))

#Gradient of By over y 
def dBy_dy_H1(x,y,z):
    f = lambda theta : -(N * mu * I2 * a/(4*np.pi)) * (3*(y-a*np.sin(theta))*(z-(e2/2))*np.cos(theta))/(((x-a*np.cos(theta))**2 + (y-a*np.sin(theta))**2 + (z-(e2/2))**2)**(5/2))
    res = integer.quad(f,-np.pi,np.pi)[0]
    return(res)
def dBy_dy_H2(x,y,z):
    f = lambda theta : -(N * mu * I2 * a/(4*np.pi)) * (3*(y-a*np.sin(theta))*(z+(e2/2))*np.cos(theta))/(((x-a*np.cos(theta))**2 + (y-a*np.sin(theta))**2 + (z+(e2/2))**2)**(5/2))
    res = integer.quad(f,-np.pi,np.pi)[0]
    return(res)
def dBy_dy_H(x,y,z):
    return(dBy_dy_H1(x,y,z) + alpha*dBy_dy_H2(x,y,z))

#Gradient of By over z 
def dBy_dz_H1(x,y,z):
    f = lambda theta : (N * mu * I2 * a/(4*np.pi)) * ((np.sin(theta)*((x-a*np.cos(theta))**2 + (y-np.sin(theta))**2 + (z-(e2/2))**2)) - 3 * ((z-(e2/2))**2)*np.sin(theta))/(((x-a*np.cos(theta))**2 + (y-a*np.sin(theta))**2 + (z-(e2/2))**2)**(5/2))
    res = integer.quad(f,-np.pi,np.pi)[0]
    return(res)
def dBy_dz_H2(x,y,z):
    f = lambda theta : (N * mu * I2 * a/(4*np.pi)) * ((np.sin(theta)*((x-a*np.cos(theta))**2 + (y-np.sin(theta))**2 + (z+(e2/2))**2)) - 3 * ((z+(e2/2))**2)*np.sin(theta))/(((x-a*np.cos(theta))**2 + (y-a*np.sin(theta))**2 + (z+(e2/2))**2)**(5/2))
    res = integer.quad(f,-np.pi,np.pi)[0]
    return(res)
def dBy_dz_H(x,y,z):
    return(dBy_dz_H1(x,y,z) + alpha*dBy_dz_H2(x,y,z))

#Gradient of Bz over x 
def dBz_dx_H1(x,y,z):
    f = lambda theta : -(N * mu * I2 * a/(4*np.pi)) * ((np.cos(theta)*((x-a*np.cos(theta))**2 + (y-a*np.sin(theta))**2 + (z-(e2/2))**2)) + 3*(a-x*np.cos(theta)-y*np.sin(theta))*(x-a*np.cos(theta)))/(((x-a*np.cos(theta))**2 + (y-a*np.sin(theta))**2 + (z-(e2/2))**2)**(5/2))
    res = integer.quad(f,-np.pi,np.pi)[0]
    return(res)
def dBz_dx_H2(x,y,z):
    f = lambda theta : -(N * mu * I2 * a/(4*np.pi)) * ((np.cos(theta)*((x-a*np.cos(theta))**2 + (y-a*np.sin(theta))**2 + (z+(e2/2))**2)) + 3*(a-x*np.cos(theta)-y*np.sin(theta))*(x-a*np.cos(theta)))/(((x-a*np.cos(theta))**2 + (y-a*np.sin(theta))**2 + (z+(e2/2))**2)**(5/2))
    res = integer.quad(f,-np.pi,np.pi)[0]
    return(res)
def dBz_dx_H(x,y,z):
    return(dBz_dx_H1(x,y,z) + alpha*dBz_dx_H2(x,y,z))
 
#Gradient of Bz over y
def dBz_dy_H1(x,y,z):
    f = lambda theta : -(N * mu * I2 * a/(4*np.pi)) * ((np.sin(theta)*((x-a*np.cos(theta))**2 + (y-a*np.sin(theta))**2 + (z-(e2/2))**2)) + 3*(a-x*np.cos(theta)-y*np.sin(theta))*(y-a*np.sin(theta)))/(((x-a*np.cos(theta))**2 + (y-a*np.sin(theta))**2 + (z-(e2/2))**2)**(5/2))
    res = integer.quad(f,-np.pi,np.pi)[0]
    return(res)
def dBz_dy_H2(x,y,z):
    f = lambda theta : -(N * mu * I2 * a/(4*np.pi)) * ((np.sin(theta)*((x-a*np.cos(theta))**2 + (y-a*np.sin(theta))**2 + (z+(e2/2))**2)) + 3*(a-x*np.cos(theta)-y*np.sin(theta))*(y-a*np.sin(theta)))/(((x-a*np.cos(theta))**2 + (y-a*np.sin(theta))**2 + (z+(e2/2))**2)**(5/2))
    res = integer.quad(f,-np.pi,np.pi)[0]
    return(res)
def dBz_dy_H(x,y,z):
    return(dBz_dy_H1(x,y,z) + alpha*dBz_dy_H2(x,y,z))

#Gradient of Bz over z 
def dBz_dz_H1(x,y,z):
    f = lambda theta : (N * mu * I2 * a/(4*np.pi)) * (3*(z-(e2/2))*(x*np.cos(theta) + y*np.sin(theta) - a))/(((x-a*np.cos(theta))**2 + (y-a*np.sin(theta))**2 + (z-(e2/2))**2)**(5/2))
    res = integer.quad(f,-np.pi,np.pi)[0]
    return(res)
def dBz_dz_H2(x,y,z):
    f = lambda theta : (N * mu * I2 * a/(4*np.pi)) * (3*(z+(e2/2))*(x*np.cos(theta) + y*np.sin(theta) - a))/(((x-a*np.cos(theta))**2 + (y-a*np.sin(theta))**2 + (z+(e2/2))**2)**(5/2))
    res = integer.quad(f,-np.pi,np.pi)[0]
    return(res)
def dBz_dz_H(x,y,z):
    return(dBz_dz_H1(x,y,z) + alpha*dBz_dz_H2(x,y,z))


#show the plot of dBx/dx
def afficher_dBx_dx_H():

    fig = plt.figure()
    ax1 = fig.add_axes([0.3, 0.1, 0.5, 0.8])
    P = np.linspace(-e2, e2, 120)
    Lx = []
    Ly = []
    Lz = []
    for p in P:
        Lx.append(dBx_dx_H(p,0,0))
        Ly.append(dBx_dx_H(0,p,0))
        Lz.append(dBx_dx_H(0,0,p))
        Yx = np.array(Lx)
        Yy = np.array(Ly)
        Yz = np.array(Lz)
    ax1.set_xlabel("x")
    ax1.set_ylabel("dBx_dx")
    ax1.set_title("Helmholtz: dBx_dx en fonction de x")
    ax1.plot(P, Yx)

    ax2 = fig.add_axes([1.0, 0.1, 0.6, 0.8])
    ax2.set_xlabel("y")
    ax2.set_ylabel("dBx_dx")
    ax2.set_title("Helmholtz: dBx_dx en fonction de y")
    ax2.plot(P, Yy)

    ax2 = fig.add_axes([1.8, 0.1, 0.6, 0.8])
    ax2.set_xlabel("z")
    ax2.set_ylabel("dBx_dx")
    ax2.set_title("Helmholtz: dBx_dx en fonction de z")
    ax2.plot(P, Yz)
    
    plt.show()

#show the plot of dBx/dy
def afficher_dBx_dy_H():

    fig = plt.figure()
    ax1 = fig.add_axes([0.3, 0.1, 0.5, 0.8])
    P = np.linspace(-e2, e2, 120)
    Lx = []
    Ly = []
    Lz = []
    for p in P:
        Lx.append(dBx_dy_H(p,0,0))
        Ly.append(dBx_dy_H(0,p,0))
        Lz.append(dBx_dy_H(0,0,p))
        Yx = np.array(Lx)
        Yy = np.array(Ly)
        Yz = np.array(Lz)
    ax1.set_xlabel("x")
    ax1.set_ylabel("dBx_dy")
    ax1.set_title("Helmholtz: dBx_dy en fonction de x")
    ax1.plot(P, Yx)

    ax2 = fig.add_axes([1.0, 0.1, 0.6, 0.8])
    ax2.set_xlabel("y")
    ax2.set_ylabel("dBx_dy")
    ax2.set_title("Helmholtz: dBx_dy en fonction de y")
    ax2.plot(P, Yy)

    ax2 = fig.add_axes([1.8, 0.1, 0.6, 0.8])
    ax2.set_xlabel("z")
    ax2.set_ylabel("dBx_dy")
    ax2.set_title("Helmholtz: dBx_dy en fonction de z")
    ax2.plot(P, Yz)
    
    plt.show()

#show the plot of dBx/dz
def afficher_dBx_dz_H():

    fig = plt.figure()
    ax1 = fig.add_axes([0.3, 0.1, 0.5, 0.8])
    P = np.linspace(-e2, e2, 120)
    Lx = []
    Ly = []
    Lz = []
    for p in P:
        Lx.append(dBx_dz_H(p,0,0))
        Ly.append(dBx_dz_H(0,p,0))
        Lz.append(dBx_dz_H(0,0,p))
        Yx = np.array(Lx)
        Yy = np.array(Ly)
        Yz = np.array(Lz)
    ax1.set_xlabel("x")
    ax1.set_ylabel("dBx_dz")
    ax1.set_title("Helmholtz: dBx_dz en fonction de x")
    ax1.plot(P, Yx)

    ax2 = fig.add_axes([1.0, 0.1, 0.6, 0.8])
    ax2.set_xlabel("y")
    ax2.set_ylabel("dBx_dz")
    ax2.set_ylim(-30,30)
    ax2.set_title("Helmholtz: dBx_dz en fonction de y")
    ax2.plot(P, Yy)

    ax2 = fig.add_axes([1.8, 0.1, 0.6, 0.8])
    ax2.set_xlabel("z")
    ax2.set_ylabel("dBx_dz")
    ax2.set_ylim(-30,30)
    ax2.set_title("Helmholtz: dBx_dz en fonction de z")
    ax2.plot(P, Yz)
    
    plt.show()

#show the plot of dBy/dx
def afficher_dBy_dx_H():

    fig = plt.figure()
    ax1 = fig.add_axes([0.3, 0.1, 0.5, 0.8])
    P = np.linspace(-e2, e2, 120)
    Lx = []
    Ly = []
    Lz = []
    for p in P:
        Lx.append(dBy_dx_H(p,0,0))
        Ly.append(dBy_dx_H(0,p,0))
        Lz.append(dBy_dx_H(0,0,p))
        Yx = np.array(Lx)
        Yy = np.array(Ly)
        Yz = np.array(Lz)
    ax1.set_xlabel("x")
    ax1.set_ylabel("dBy_dx")
    ax1.set_title("Helmholtz: dBy_dx en fonction de x")
    ax1.plot(P, Yx)

    ax2 = fig.add_axes([1.0, 0.1, 0.6, 0.8])
    ax2.set_xlabel("y")
    ax2.set_ylabel("dBy_dx")
    ax2.set_title("Helmholtz: dBy_dx en fonction de y")
    ax2.plot(P, Yy)

    ax2 = fig.add_axes([1.8, 0.1, 0.6, 0.8])
    ax2.set_xlabel("z")
    ax2.set_ylabel("dBy_dx")
    ax2.set_title("Helmholtz: dBy_dx en fonction de z")
    ax2.plot(P, Yz)
    
    plt.show()

#show the plot of dBy/dy    
def afficher_dBy_dy_H():

    fig = plt.figure()
    ax1 = fig.add_axes([0.3, 0.1, 0.5, 0.8])
    P = np.linspace(-e2, e2, 120)
    Lx = []
    Ly = []
    Lz = []
    for p in P:
        Lx.append(dBy_dy_H(p,0,0))
        Ly.append(dBy_dy_H(0,p,0))
        Lz.append(dBy_dy_H(0,0,p))
        Yx = np.array(Lx)
        Yy = np.array(Ly)
        Yz = np.array(Lz)
    ax1.set_xlabel("x")
    ax1.set_ylabel("dBy_dy")
    ax1.set_title("Helmholtz: dBy_dy en fonction de x")
    ax1.plot(P, Yx)

    ax2 = fig.add_axes([1.0, 0.1, 0.6, 0.8])
    ax2.set_xlabel("y")
    ax2.set_ylabel("dBy_dy")
    ax2.set_title("Helmholtz: dBy_dy en fonction de y")
    ax2.plot(P, Yy)

    ax2 = fig.add_axes([1.8, 0.1, 0.6, 0.8])
    ax2.set_xlabel("z")
    ax2.set_ylabel("dBy_dy")
    ax2.set_title("Helmholtz: dBy_dy en fonction de z")
    ax2.plot(P, Yz)
    
    plt.show()    

#show the plot of dBy/dz
def afficher_dBy_dz_H():

    fig = plt.figure()
    ax1 = fig.add_axes([0.3, 0.1, 0.5, 0.8])
    P = np.linspace(-e2, e2, 120)
    Lx = []
    Ly = []
    Lz = []
    for p in P:
        Lx.append(dBy_dz_H(p,0,0))
        Ly.append(dBy_dz_H(0,p,0))
        Lz.append(dBy_dz_H(0,0,p))
        Yx = np.array(Lx)
        Yy = np.array(Ly)
        Yz = np.array(Lz)
    ax1.set_xlabel("x")
    ax1.set_ylabel("dBy_dz")
    ax1.set_title("Helmholtz: dBy_dz en fonction de x")
    ax1.plot(P, Yx)

    ax2 = fig.add_axes([1.0, 0.1, 0.6, 0.8])
    ax2.set_xlabel("y")
    ax2.set_ylabel("dBy_dz")
    ax2.set_title("Helmholtz: dBy_dz en fonction de y")
    ax2.plot(P, Yy)

    ax2 = fig.add_axes([1.8, 0.1, 0.6, 0.8])
    ax2.set_xlabel("z")
    ax2.set_ylabel("dBy_dz")
    #ax2.set_ylim(-0.001,0.001)
    ax2.set_title("Helmholtz: dBy_dz en fonction de z")
    ax2.plot(P, Yz)
    
    plt.show()

#show the plot of dBz/dx
def afficher_dBz_dx_H():

    fig = plt.figure()
    ax1 = fig.add_axes([0.3, 0.1, 0.5, 0.8])
    P = np.linspace(-e2, e2, 120)
    Lx = []
    Ly = []
    Lz = []
    for p in P:
        Lx.append(dBz_dx_H(p,0,0))
        Ly.append(dBz_dx_H(0,p,0))
        Lz.append(dBz_dx_H(0,0,p))
        Yx = np.array(Lx)
        Yy = np.array(Ly)
        Yz = np.array(Lz)
    ax1.set_xlabel("x")
    ax1.set_ylabel("dBz_dx")
    ax1.set_title("Helmholtz: dBz_dx en fonction de x")
    ax1.plot(P, Yx)

    ax2 = fig.add_axes([1.0, 0.1, 0.6, 0.8])
    ax2.set_xlabel("y")
    ax2.set_ylabel("dBz_dx")
    ax2.set_ylim(-0.9,0.9)
    ax2.set_title("Helmholtz: dBz_dx en fonction de y")
    ax2.plot(P, Yy)

    ax2 = fig.add_axes([1.8, 0.1, 0.6, 0.8])
    ax2.set_xlabel("z")
    ax2.set_ylabel("dBz_dx")
    ax2.set_ylim(-0.9,0.9)
    ax2.set_title("Helmholtz: dBz_dx en fonction de z")
    ax2.plot(P, Yz)
    
    plt.show()

#show the plot of dBz/dy    
def afficher_dBz_dy_H():

    fig = plt.figure()
    ax1 = fig.add_axes([0.3, 0.1, 0.5, 0.8])
    P = np.linspace(-e2, e2, 120)
    Lx = []
    Ly = []
    Lz = []
    for p in P:
        Lx.append(dBz_dy_H(p,0,0))
        Ly.append(dBz_dy_H(0,p,0))
        Lz.append(dBz_dy_H(0,0,p))
        Yx = np.array(Lx)
        Yy = np.array(Ly)
        Yz = np.array(Lz)
    ax1.set_xlabel("x")
    ax1.set_ylabel("dBz_dy")
    ax1.set_title("Helmholtz: dBz_dy en fonction de x")
    ax1.plot(P, Yx)

    ax2 = fig.add_axes([1.0, 0.1, 0.6, 0.8])
    ax2.set_xlabel("y")
    ax2.set_ylabel("dBz_dy")
    ax2.set_title("Helmholtz: dBz_dy en fonction de y")
    ax2.plot(P, Yy)

    ax2 = fig.add_axes([1.8, 0.1, 0.6, 0.8])
    ax2.set_xlabel("z")
    ax2.set_ylabel("dBz_dy")
    ax2.set_title("Helmholtz: dBz_dy en fonction de z")
    ax2.plot(P, Yz)
    
    plt.show()

#show the plot of dBz/dz
def afficher_dBz_dz_H():

    fig = plt.figure()
    ax1 = fig.add_axes([0.3, 0.1, 0.5, 0.8])
    P = np.linspace(-e2, e2, 120)
    Lx = []
    Ly = []
    Lz = []
    for p in P:
        Lx.append(dBz_dz_H(p,0,0))
        Ly.append(dBz_dz_H(0,p,0))
        Lz.append(dBz_dz_H(0,0,p))
        Yx = np.array(Lx)
        Yy = np.array(Ly)
        Yz = np.array(Lz)
    ax1.set_xlabel("x")
    ax1.set_ylabel("dBz_dz")
    ax1.set_title("Helmholtz: dBz_dz en fonction de x")
    ax1.plot(P, Yx)

    ax2 = fig.add_axes([1.0, 0.1, 0.6, 0.8])
    ax2.set_xlabel("y")
    ax2.set_ylabel("dBz_dz")
    ax2.set_title("Helmholtz: dBz_dz en fonction de y")
    ax2.plot(P, Yy)

    ax2 = fig.add_axes([1.8, 0.1, 0.6, 0.8])
    ax2.set_xlabel("z")
    ax2.set_ylabel("dBz_dz")
    ax2.set_title("Helmholtz: dBz_dz en fonction de z")
    ax2.plot(P, Yz)
    
    plt.show()
 
    
#show the plot of Bx    
def afficher_Bx_H():

    fig = plt.figure()
    ax1 = fig.add_axes([0.3, 0.1, 0.5, 0.8])
    P = np.linspace(-e2, e2, 120)
    Lx = []
    Ly = []
    Lz = []
    for p in P:
        Lx.append(Bx_H(p,0,0))
        Ly.append(Bx_H(0,p,0))
        Lz.append(Bx_H(0,0,p))
        Yx = np.array(Lx)
        Yy = np.array(Ly)
        Yz = np.array(Lz)
    ax1.set_xlabel("x")
    ax1.set_ylabel("Bx")
    ax1.set_title("Helmholtz: Bx en fonction de x")
    ax1.plot(P, Yx)

    ax2 = fig.add_axes([1.0, 0.1, 0.6, 0.8])
    ax2.set_xlabel("y")
    ax2.set_ylabel("Bx")
    ax2.set_title("Helmholtz: Bx en fonction de y")
    ax2.plot(P, Yy)

    ax2 = fig.add_axes([1.8, 0.1, 0.6, 0.8])
    ax2.set_xlabel("z")
    ax2.set_ylabel("Bx")
    ax2.set_ylim(-0.04,0.04)
    ax2.set_title("Helmholtz: Bx en fonction de z")
    ax2.plot(P, Yz)
    
    plt.show()


 
#show the plot of By    
def afficher_By_H():

    fig = plt.figure()
    ax1 = fig.add_axes([0.3, 0.1, 0.5, 0.8])
    P = np.linspace(-e2, e2, 120)
    Lx = []
    Ly = []
    Lz = []
    for p in P:
        Lx.append(By_H(p,0,0))
        Ly.append(By_H(0,p,0))
        Lz.append(By_H(0,0,p))
        Yx = np.array(Lx)
        Yy = np.array(Ly)
        Yz = np.array(Lz)
    ax1.set_xlabel("x")
    ax1.set_ylabel("By")
    ax1.set_title("Helmholtz: By en fonction de x")
    ax1.plot(P, Yx)

    ax2 = fig.add_axes([1.0, 0.1, 0.6, 0.8])
    ax2.set_xlabel("y")
    ax2.set_ylabel("By")
    ax2.set_title("Helmholtz: By en fonction de y")
    ax2.plot(P, Yy)

    ax2 = fig.add_axes([1.8, 0.1, 0.6, 0.8])
    ax2.set_xlabel("z")
    ax2.set_ylabel("By")
    ax2.set_title("Helmholtz: By en fonction de z")
    ax2.plot(P, Yz)
    
    plt.show()



#show the plot of Bz    
def afficher_Bz_H():

    fig = plt.figure()
    ax1 = fig.add_axes([0.3, 0.1, 0.5, 0.8])
    P = np.linspace(-e2, e2, 120)
    Lx = []
    Ly = []
    Lz = []
    for p in P:
        Lx.append(Bz_H(p,0,0))
        Ly.append(Bz_H(0,p,0))
        Lz.append(Bz_H(0,0,p))
        Yx = np.array(Lx)
        Yy = np.array(Ly)
        Yz = np.array(Lz)
    ax1.set_xlabel("x")
    ax1.set_ylabel("Bz")
    ax1.set_title("Helmholtz: Bz en fonction de x")
    ax1.plot(P, Yx)

    ax2 = fig.add_axes([1.0, 0.1, 0.6, 0.8])
    ax2.set_xlabel("y")
    ax2.set_ylabel("Bz")
    ax2.set_title("Helmholtz: Bz en fonction de y")
    ax2.plot(P, Yy)

    ax2 = fig.add_axes([1.8, 0.1, 0.6, 0.8])
    ax2.set_xlabel("z")
    ax2.set_ylabel("Bz")
    ax2.set_title("Helmholtz: Bz en fonction de z")
    ax2.plot(P, Yz)
    
    plt.show()


#Magnetic force generated by Helmholtz coils

def Fx_H(x,y,z):
    return(suscep*vol/mu)*(Bx_H(x,y,z) * dBx_dx_H(x,y,z) + By_H(x,y,z)*dBx_dy_H(x,y,z) + Bz_H(x,y,z)*dBx_dz_H(x,y,z))
def Fy_H(x,y,z):
    return(suscep*vol/mu)*(Bx_H(x,y,z) * dBy_dx_H(x,y,z) + By_H(x,y,z)*dBy_dy_H(x,y,z) + Bz_H(x,y,z)*dBy_dz_H(x,y,z))
def Fz_H(x,y,z):
    return(suscep*vol/mu)*(Bx_H(x,y,z) * dBz_dx_H(x,y,z) + By_H(x,y,z)*dBz_dy_H(x,y,z) + Bz_H(x,y,z)*dBz_dz_H(x,y,z))

#show the plot of Fx
def afficher_Fx_H():

    fig = plt.figure()
    ax1 = fig.add_axes([0.3, 0.1, 0.5, 0.8])
    P = np.linspace(-e2, e2, 120)
    Lx = []
    Ly = []
    Lz = []
    #pos_axe = 0   #Position sur l'axe z
    for p in P:
        Lx.append(Fx_H(p,0,0))
        Ly.append(Fx_H(0,p,0))
        Lz.append(Fx_H(0,0,p))
    Yx = np.array(Lx)
    Yy = np.array(Ly)
    Yz = np.array(Lz)
    ax1.set_xlabel("x")
    ax1.set_ylabel("Fx")
    ax1.set_title("H: Fx en fonction de x")
    ax1.plot(P, Yx)

    ax2 = fig.add_axes([1.0, 0.1, 0.6, 0.8])
    ax2.set_xlabel("y")
    ax2.set_ylabel("Fx")
    ax2.set_ylim(-4*10**(-15),4*10**(-15))
    ax2.set_title("H: Fx en fonction de y")
    ax2.plot(P, Yy)

    ax2 = fig.add_axes([1.8, 0.1, 0.6, 0.8])
    ax2.set_xlabel("z")
    ax2.set_ylabel("Fx")
    ax2.set_ylim(-4*10**(-15),4*10**(-15))
    ax2.set_title("H: Fx en fonction de z")
    ax2.plot(P, Yz)

    plt.show()


#show the plot of Fy
def afficher_Fy_H():

    fig = plt.figure()
    ax1 = fig.add_axes([0.3, 0.1, 0.5, 0.8])
    P = np.linspace(-e2, e2, 120)
    Lx = []
    Ly = []
    Lz = []
    #pos_axe = 0   #Position sur l'axe z
    for p in P:
        Lx.append(Fy_H(p,0,0))
        Ly.append(Fy_H(0,p,0))
        Lz.append(Fy_H(0,0,p))
    Yx = np.array(Lx)
    Yy = np.array(Ly)
    Yz = np.array(Lz)
    ax1.set_xlabel("x")
    ax1.set_ylabel("Fy")
    ax1.set_ylim(-2*10**(-14),2*10**(-14))
    ax1.set_title("H: Fy en fonction de x")
    ax1.plot(P, Yx)

    ax2 = fig.add_axes([1.0, 0.1, 0.6, 0.8])
    ax2.set_xlabel("y")
    ax2.set_ylabel("Fy")
    ax2.set_title("H: Fy en fonction de y")
    ax2.plot(P, Yy)

    ax2 = fig.add_axes([1.8, 0.1, 0.6, 0.8])
    ax2.set_xlabel("z")
    ax2.set_ylabel("Fy")
    ax2.set_ylim(-2*10**(-14),2*10**(-14))
    ax2.set_title("H: Fy en fonction de z")
    ax2.plot(P, Yz)

    plt.show()


#show the plot of Fz
def afficher_Fz_H():

    fig = plt.figure()
    ax1 = fig.add_axes([0.3, 0.1, 0.5, 0.8])
    P = np.linspace(-2*e2, 2*e2, 120)
    Lx = []
    Ly = []
    Lz = []
    #pos_axe = 0   #Position sur l'axe z
    for p in P:
        Lx.append(Fz_H(p,0,0))
        Ly.append(Fz_H(0,p,0))
        Lz.append(Fz_H(0,0,p))
    Yx = np.array(Lx)
    Yy = np.array(Ly)
    Yz = np.array(Lz)
    ax1.set_xlabel("x")
    ax1.set_ylabel("Fz")
    ax1.set_title("H: Fz en fonction de x")
    ax1.plot(P, Yx)

    ax2 = fig.add_axes([1.0, 0.1, 0.6, 0.8])
    ax2.set_xlabel("y")
    ax2.set_ylabel("Fz")
    ax2.set_title("H: Fz en fonction de y")
    ax2.plot(P, Yy)

    ax2 = fig.add_axes([1.8, 0.1, 0.6, 0.8])
    ax2.set_xlabel("z")
    ax2.set_ylabel("Fz")
    ax2.set_title("H: Fz en fonction de z")
    ax2.plot(P, Yz)

    plt.show()



