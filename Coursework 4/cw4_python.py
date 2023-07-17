import numpy as np
import math
import matplotlib.pyplot as plt
#from tqdm import tqdm
from mpl_toolkits.mplot3d import axes3d
from matplotlib import cm
import cmath
from matplotlib.colors import colorConverter

# ============================================================================================
# Question 2

def q3(h=1e-4, tau0=1e-3, v=1., eps=0.1, L=10, tmax=1, tol=1e-5):
    """Solution for linear IBVP"""
    
    # Setting up solution grid
    M = int(math.ceil(L/h))
    U = np.zeros((M+1, 1))
    
    xspan = np.linspace(0, L, M+1)
    
    # ----- Initial condition -----
    initial_condition = lambda x: np.cos(2*np.pi*x/L) 
    initial_condition = np.vectorize(initial_condition)
    U[:,0] = initial_condition(xspan)
    
    # ----- Finite Difference matrix -----
    A = np.zeros((M+1,M+1))
    A += np.diag(np.repeat(eps/(h**2)-v/(2*h),M), 1) 
    A += np.diag(np.repeat(-2*eps/(h**2),M+1)) 
    A += np.diag(np.repeat(eps/(h**2)+v/(2*h),M), -1)
   
    # ----- Boundary condition -----
    A[0,1], A[-1,-2] = 2*eps/(h**2), 2*eps/(h**2)
    
    # ----- RK4 coeffs -----
    a11, a22 = 1/4, 1/4
    a12, a21 = (math.sqrt(3)-2)/(4*math.sqrt(3)), (math.sqrt(3)+2)/(4*math.sqrt(3))
    b1, b2 = 1/2, 1/2
    
    
    def get_timestep(u_RK3, u_RK4, tau):
        """Returns next time step size"""
        
        R = u_RK4 - u_RK3
        tau_new = tau*(tol/np.sqrt(sum(R**2)))**(1/4)
        
        return tau_new
    
    def f(u):
        return A @ u
    
    def RK3(u, tau):
        """Third order runge kutta solver"""
        k1 = f(u)
        k2 = f(u+tau*k1/3)
        k3 = f(u+2*tau*k2)
        
        return u + tau * (k1+3*k3)/4
    
    
    def RK4(u, tau):
        """Third order runge kutta solver"""
        
        k2 = np.linalg.inv((np.eye(M+1)-tau*a11*A)@(np.eye(M+1)-tau*a22*A)-tau**2*a12*a21*A@A) @ (tau*a21*A+(np.eye(M+1)-tau*a11*A))@A@u
        k1 = np.linalg.inv(np.eye(M+1)-tau*a11*A) @ (A@u+tau*a12*k2)
        un_plus_tau = u + tau*(b1*k1 + b2*k2)
        
        return un_plus_tau
        
    
    # ---- Run ----     
    tau = 1e-5
    running_time = tau
    time_step_list = [0]

    while running_time < tmax:
        
        time_step_list.append(running_time)
        
        un_plus_RK3= RK3(U[:,-1], tau)                         # RK3 SOLUTION
        un_plus_RK4= RK4(U[:,-1], tau)                         # RK4 SOLUTION
        U = np.hstack((U, un_plus_RK3.reshape((M+1,1))))
        
        tau = get_timestep(un_plus_RK3, un_plus_RK4, tau)   # Next time step
        running_time += tau

    # Last time step
    tau = tmax - running_time
    un_plus = RK3(U[:,-1], tau)
    U = np.hstack((U, un_plus_RK3.reshape((M+1,1))))
    
    time_step_list.append(tmax)
    time_steps = np.array(time_step_list)
    
    return U, time_steps, A

def plot2(U, M, N, tmax=400, L=100, az=0):
    xspan = np.linspace(0, L, M+1)
    tspan = np.linspace(0, tmax, N+1)
    t, x = np.meshgrid(tspan, xspan)
    fig, ax = plt.subplots(1, 1, subplot_kw={'projection': '3d'}, figsize=(18,10))
    ax.plot_surface(x,t,U, cmap=cm.jet)
    ax.view_init(azim=az)
    ax.set_xlabel('Space')
    ax.set_ylabel('Time')
    plt.show()
    
print('===================== QUESTION 2 =====================')
U,time, a = q3(h=0.025, tau0=0.05, v=1., eps=0.1, L=10, tmax=1)
plot2(U, U.shape[0]-1, U.shape[1]-1, tmax=1, L=10, az=130)

# ============================================================================================
# Question 3

def q4(h=1e-4, tau=1e-3, tmax=400, L=100):
    """Solution for Camassa-Holm IBVP"""
    
    # Setting up solution grid
    M = int(math.ceil(L/h))
    N = int(math.ceil(tmax/tau))
    U = np.zeros((M+1, N+1))
    
    xspan = np.linspace(0, L, M+1)
    tspan = np.linspace(0, tmax, N+1)
    
    # ----- Initial condition -----
    initial_condition = lambda x: np.exp(-5*(x-L/3)**2)/10 - np.exp(-5*(x-2*L/3)**2)/10
    initial_condition = np.vectorize(initial_condition)
    U[:,0] = initial_condition(xspan)
    
    
    # ---- Finite Difference matrix ----
    A = np.zeros((M+1,M+1))
    A += np.diag(np.repeat(-1/(h**2), M), 1) 
    A += np.diag(np.repeat(1+2/(h**2),M+1)) 
    A += np.diag(np.repeat(-1/(h**2), M), -1)
    
    # ----- Boundary condition -----
    A[0,-2] = -1/(h**2)
    A[-1,1] = -1/(h**2)
    
    A_inv = np.linalg.inv(A)

        
    def f(u):
        
        u_ext = np.hstack((u[-3:-1], u, u[1:3]))
        up2 = u_ext[4:]
        up1 = u_ext[3:-1]
        um1 = u_ext[1:-3]
        um2 = u_ext[:-4]
        
        part1 = - (u - (up1-2*u+um1)/h**2)*((up1-um1)/(2*h)) - (up1**2-um1**2)/(2*h)
        part2 = ( up1*(up2-2*up1+u) - um1*(u-2*um1+um2) )/(2*h**3)
        
        return A_inv @ (part1 + part2) 
        
    
    def f_prime(u):
        """Returns A^{-1}G' as in the report"""
        
        u_ext =  np.hstack((u[-3:-1], u, u[1:3]))
        
        up2 = u_ext[4:]
        up1 = u_ext[3:-1]
        um1 = u_ext[1:-3]
        um2 = u_ext[:-4]
        
        aux = (up1 - um1)/(2*h)
        
        u_i_j = - aux * (1 + h**(-2))
        u_ip1_j = (-h**2*(2*up1+u) - 2*up1+up2-u)/(2*h**3)
        u_im1_j = (h**2*(2*um1+u) + 2*um1-um2+u)/(2*h**3)
        u_ip2_j = up1/(2*h**3)
        u_im2_j = -um1/(2*h**3)
        
        mat = np.diag(u_i_j) + np.diag(u_ip1_j[:-1], 1) + np.diag(u_im1_j[1:], -1) 
        mat += np.diag(u_ip2_j[:-2], 2) + np.diag(u_im2_j[2:], 2)
                       
        mat[-2,1], mat[-1,2] = u_ip2_j[-2:]
        mat[0,-3], mat[1,-2] = u_im2_j[:2]
        mat[0,-2], mat[-1,1] = u_im1_j[1], u_ip1_j[-1]
        
        return A_inv @ mat
        
      
    def newton(un, k1, tolerance=1e-8):
        """
        Newton method implementation for: k2 - f(u+tau*(k1/2+k2/2))
        """
        # Initilisation
        n_iter, tol = 0, np.inf
        k2_old = f(un + tau*k1)
        
        while tol > tolerance:
            
            F_vec = k2_old - f(un + tau*(k1+k2_old)/2)
            F_inv_prime = np.linalg.inv(np.eye(M+1) - f_prime(un + tau*(k1+k2_old)/2) * tau/2)
            k2_new = k2_old - F_inv_prime @ F_vec
            tol = np.sqrt(sum((k2_new-k2_old)**2)) / np.sqrt(sum((k2_new)**2))
            k2_old = k2_new
            n_iter +=1
                
        return k2_new, n_iter
    
    
    # ---- Crack-Nicholson ----
    
    def crack_nicholson(un):
        
        k1 = f(un)
        k2, it = newton(un, k1)

        return un + tau*(k1+k2)/2, it  
   
    # ---- Run ----
    iter_list = []
    
    #for i in tqdm(range(1,N+1)):
    for i in range(1,N+1):
        
        mt, it = crack_nicholson(U[:,i-1])
        iter_list.append(it)
        U[:,i] = mt
        
    return U, iter_list

def plot(U, h, tau, tmax=400, L=100):
    M = int(math.ceil(L/h))
    N = int(math.ceil(tmax/tau))
    xspan = np.linspace(0, L, M+1)
    tspan = np.linspace(0, tmax, N+1)
    t, x = np.meshgrid(tspan, xspan)
    fig, ax = plt.subplots(1, 1, subplot_kw={'projection': '3d'}, figsize=(18,10))
    ax.plot_surface(x,t,U, cmap=cm.jet)
    ax.view_init(azim=60)
    ax.set_xlabel('Space')
    ax.set_ylabel('Time')
    plt.show()
    
    
print('===================== QUESTION 3 =====================')
h = 0.4
tau = 0.4
U, _ = q4(h=h, tau=tau, tmax=400, L=100)
plot(U, 0.4,0.4)