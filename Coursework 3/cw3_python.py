import numpy as np 
import matplotlib.pyplot as plt
import math
from matplotlib.colors import colorConverter
from mpl_toolkits import mplot3d
import cmath
import time

# ====================================== QUESTION 1 ======================================

def q1():
    """
    Function that plot region of absolute stabity for 
    explicit, implicit and PECE methods
    """
    
    # Setting linspace for plotting
    s = np.linspace(0,2*cmath.pi,1000)
    unit = np.exp(1j*s)
    
    # Computing real and imag part of h(s) for explicit method
    h_ex = 24*(unit**3-(3/2)*unit**2+unit/2)/(11+41*unit**2-40*unit)
    re_ex = h_ex.real
    im_ex = h_ex.imag

    # Computing real and imag part of h(s) for implicit method
    h_im = 72*(unit**3-(3/2)*unit**2+unit/2)/(-7+40*unit**3+3*unit**2)
    re_im = h_im.real
    im_im = h_im.imag

    # Plotting
    fig, (ax1,ax2) = plt.subplots(1,2,figsize=(18,10), constrained_layout=True)
    ax1.plot(re_ex, im_ex, color='black')
    ax1.grid()
    ax2.plot(re_im, im_im, color='black')
    ax2.grid()
    ax1.spines['left'].set_position('zero')
    ax1.spines['right'].set_color('none')
    ax1.spines['bottom'].set_position('zero')
    ax1.spines['top'].set_color('none')
    ax2.spines['left'].set_position('zero')
    ax2.spines['right'].set_color('none')
    ax2.spines['bottom'].set_position('zero')
    ax2.spines['top'].set_color('none')
    ax1.fill(re_ex, im_ex, color='red',alpha=0.2)
    ax2.fill(re_im, im_im, color='white')
    ax2.set_facecolor(color = colorConverter.to_rgba('red', 0.2))
    ax1.set_title('Absolute stability region for explicit scheme', fontsize=18)
    ax2.set_title('Absolute stability region for implicit scheme', fontsize=18)
    plt.show()
    
    # PECE method
    
    # Finiding roots h_1/2(s)
    a = -(205/216)*unit**2 + (25/27)*unit - 55/216
    b = -(7/8)*unit**2 + (5/18)*unit + 7/72
    c = unit/2 -3*(unit**2)/2+unit**3

    h1 = (-b+(b**2-4*a*c)**(1/2))/(2*a)
    h2 = (-b-(b**2-4*a*c)**(1/2))/(2*a)

    # Computing real and imag part of h_1/2(s) for PECE method
    re_pece1 = h1.real
    re_pece2 = h2.real
    im_pece1 = h1.imag
    im_pece2 = h2.imag
    
    # Plotting
    fig, ax = plt.subplots(1,1,figsize=(18,10), constrained_layout=True)
    ax.plot(re_pece1, im_pece1, color='red', label='h1')
    ax.plot(re_pece2, im_pece2, color='blue', label='h2')
    ax.grid(), ax.legend(fontsize=16, loc='upper left')
    ax.spines['left'].set_position('zero')
    ax.spines['right'].set_color('none')
    ax.spines['bottom'].set_position('zero')
    ax.spines['top'].set_color('none')
    ax.set_title('Absolute stability region for PECE scheme', fontsize=20)
    ax.scatter(-1.5, 0.08)
    plt.axis()
    plt.show()  
    
# ====================================== QUESTION 2 ======================================

class q2:
    
    def __init__(self, h=1e-3):
        
        # initializing parameters
        self.t0 = 0
        self.tN = 100
        self.h = h
    
    @staticmethod
    def f(point):
        
        x, y, z = point[0], point[1], point[2]
        
        x_prime = -0.04*x + 10**4*(y*z)
        y_prime = 0.04*x - 10**4*(y*z) - 3*10**7*(y**2) 
        z_prime = 3*10**7*(y**2)
        
        return np.array([x_prime, y_prime, z_prime])
        
    
    def predict(self, x):
        """
        Returns x_{n+1} = (3/2)*x_n - (1/2)*x_{n-1} + (h/24)*(41*f(x_n) - 40*f(x_{n-1}) + 11*f(x_{n-2})) (predictor)
        """
        # Extracting values
        xn = x[:,-1]
        xnminus1 = x[:,-2]
        xnminus2 = x[:,-3]
        
        xnplus1 = (3/2)*xn - (1/2)*xnminus1 + (self.h/24)*(41*self.f(xn) - 40*self.f(xnminus1) + 11*self.f(xnminus2))
        
        return xnplus1
    
    
    def correct(self, x_hat_nplus1, x):
        """
        Returns x_{n+1} = (3/2)*x_n - (1/2)*x_{n-1} + (h/72)*(40*f(x_{n+1}_hat) +3*f(x_n) -7*f(x_{n-2})) (corrector)
        """
        # Extracting values
        xn = x[:,-1]
        xnminus1 = x[:,-2]
        xnminus2 = x[:,-3]
        
        xnplus1 = (3/2)*xn - (1/2)*xnminus1 + (self.h/72)*(40*self.f(x_hat_nplus1) + 3*self.f(xn) -7*self.f(xnminus2))
        
        return xnplus1
        
        
    def compute(self, plot = True, return_sol=False):
        """
        Function that implements the numerical scheme
        """
        
        steps = int(math.ceil(self.tN / self.h))
        
        # Initializise solution array
        sol = np.zeros((3, steps)) 
        
        # Stacking initial values
        sol = np.hstack((np.array([1,0,0]).reshape(3,1), sol))
        t = self.t0

        # Two forward Eulers
        sol[:,1] = sol[:,0] + self.h * self.f(sol[:,0])
        t += self.h
        sol[:,2] = sol[:,1] + self.h * self.f(sol[:,1])
        t += self.h
        i = 2
        
        while t < self.tN:
            
            # Evaluate f and predict
            x_hat_nplus1 = self.predict(sol[:,i-2:i+1])
            
            # Evaluate f and correct 
            sol[:, i+1] = self.correct(x_hat_nplus1, sol[:,i-2:i+1])
            
            # Update index and t
            t += self.h
            i += 1
            
        # plotting  
        if plot:
            
            t_range = np.array(range(sol.shape[1]))*self.h
            fig, ax = plt.subplots(1,1,figsize=(18,10))
            ax.plot(t_range, sol[0,:], label='x', color='blue')
            ax.plot(t_range, sol[1,:]*10**4, label=r'10^4 y', color='red')
            ax.plot(t_range, sol[2,:], label='z', color='purple')
            ax.set_xlabel('Time')
            ax.legend(fontsize=16), ax.grid()
            ax.set_title('PECE method for Robertson chemical reaction', fontsize=18)
            plt.show()
        
        if return_sol:
            return sol
        
# ====================================== QUESTION 3 ======================================
    
class q3:
    
    def __init__(self, h=1e-4):
        
        # Initilaizing parameters
        self.h = h
        self.t0 = 0
        self.tN = 50
        
    
    @staticmethod
    def f(point):
        x, y, z = point[0], point[1], point[2]
        gamma = 0.87
        alpha = 1.1
        
        return np.array([y*(z-1+x**2)+gamma*x, x*(3*z+1-x**2)+gamma*y, -2*z*(alpha+x*y)])
    
    
    def implicit_scheme(self, xnplus1, x):
        """
        Returns x_{n+1} = (3/2)*x_n - (1/2)*x_{n-1} + (h/72)*(40*f(xnplus1) +3*f(x_n) -7*f(x_{n-2}))    
        """    
        xn = x[:,-1]
        xnminus1 = x[:,-2]
        xnminus2 = x[:,-3]
        
        xnplus1 = (3/2)*xn - (1/2)*xnminus1 + (self.h/72)*(40*self.f(xnplus1) + 3*self.f(xn) -7*self.f(xnminus2))
        
        return xnplus1
    
    @staticmethod
    def f_prime(point):
        x, y, z = point[0], point[1], point[2]
        gamma = 0.87
        alpha = 1.1
        
        mat = np.array([[2*x*y+gamma, z-1+x**2, y], [3*z+1-3*x**2, gamma, 3*x], [-2*z*y, -2*x*z, -2*(alpha+x*y)]])
        
        return mat
    
    def F(self, x, xnplus1):
        
        return xnplus1 - self.implicit_scheme(xnplus1, x)
    
    def F_prime(self, xnplus1):
        
        return np.eye(3) - (40*self.h/72) * self.f_prime(xnplus1)
    
    
    def fixed_point_iter(self, x, tolerance=1e-5):
        """
        Fixed point iteration method implementation
        """
        
        # Initilisation
        n_iter, tol = 0, np.inf
        xnplus1_old = x[:,-1]
        
        while tol > tolerance:
            
            xnplus1_new = self.implicit_scheme(xnplus1_old, x)
            tol = np.sqrt(sum((xnplus1_new-xnplus1_old)**2))/ np.sqrt(sum((xnplus1_new)**2))
            xnplus1_old = xnplus1_new
            n_iter += 1
            
        return xnplus1_new, n_iter
    
        
    def newton(self, x, tolerance=1e-5):
        """
        Newton method implementation
        """
        # Initilisation
        n_iter, tol = 0, np.inf
        xnplus1_old = x[:,-1]
        
        while tol > tolerance:
            
            xnplus1_new = xnplus1_old - np.linalg.inv(self.F_prime(xnplus1_old)) @ self.F(x, xnplus1_old)
            tol = np.sqrt(sum((xnplus1_new-xnplus1_old)**2)) / np.sqrt(sum((xnplus1_new)**2))
            xnplus1_old = xnplus1_new
            n_iter += 1
            
        return xnplus1_new, n_iter
        
    
    def compute(self, tolerance = 1e-7, method='fixed_point', plot = True):
        
        assert method in ['fixed_point', 'newton']
        
        start = time.time()
        steps = int(math.ceil(self.tN / self.h))
        n_iter_list = []
        
        
        # Initializise solution array
        sol = np.zeros((3, steps)) 
        
        # Stacking initial values
        sol = np.hstack((np.array([-1,0,0.5]).reshape(3,1), sol))
        
        t = self.t0

        # Two forward Eulers
        sol[:,1] = sol[:,0] + self.h * self.f(sol[:,0])
        t += self.h
        sol[:,2] = sol[:,1] + self.h * self.f(sol[:,1])
        t += self.h
        i = 2
        
        while t < self.tN:
            
            if method == 'newton': # Newton method
                sol[:,i+1], n_iter = self.newton(sol[:,i-2:i+1], tolerance=tolerance)
                
            else: # Fixed point iteration 
                sol[:,i+1], n_iter = self.fixed_point_iter(sol[:,i-2:i+1], tolerance=tolerance)
            
            # Update index and t
            t += self.h
            i += 1
            n_iter_list.append(n_iter)
            
        time_taken = time.time() - start
        self.sol = sol
        
        if plot:
            self.plot(tol=tolerance, method=method)
        
        return sol, time_taken, n_iter_list
    
    
    def plot(self, tol, method='fixed_point'):
        """ Plotting fucntion for both 3D adn against time plots"""
        
        assert self.sol is not None, 'Use compute methdo before ploting'
        
        t_range = np.array(range(self.sol.shape[1]))*self.h
        
        fig = plt.figure(figsize=(18,9), constrained_layout=True)
        ax = fig.add_subplot(1, 2, 1, projection='3d')
        ax.plot3D(self.sol[0,:], self.sol[1,:], self.sol[2,:], 'blue', alpha=0.7)
        ax.view_init(azim=240), ax.grid()
        
        ax1 = fig.add_subplot(1, 2, 2)
        ax1.plot(t_range, self.sol[0,:], label='x', color='blue')
        ax1.plot(t_range, self.sol[1,:], label='y', color='red')
        ax1.plot(t_range, self.sol[2,:], label='z', color='purple')
        ax1.set_xlabel('Time')
        ax1.legend(fontsize=16), ax1.grid()

        # plotting
        if method == 'fixed_point':
            plt.suptitle('Implicit scheme with fixed point method for Rabinovich–Fabrikant system, tol = {}'.format(tol), fontsize=18)
        else:
            plt.suptitle('Implicit scheme with Newton method for Rabinovich–Fabrikant system, tol = {}'.format(tol), fontsize=18)
        plt.show()
        
        
    def run(self, tol_list=[1e-8, 1e-13]):
             
        list1, list2 = [], []
        
        for tolerance, plt_var, in zip(tol_list, [True, False]):
            
            _, time1, n_iter_list1 = self.compute(tolerance=tolerance, method='fixed_point', plot=plt_var)
            _, time2, n_iter_list2 = self.compute(tolerance=tolerance, method='newton', plot=plt_var)
            
            list1.append(n_iter_list1)
            list2.append(n_iter_list2)
            
            print('--------------- Tolerance = {} ---------------'.format(tolerance))
            print('Fixed point iter method time = {}'.format(time1))
            print('Newton method time = {}'.format(time2))
            print('Fixed point iter method total n_iter = {}'.format(sum(np.array(n_iter_list1))))
            print('Newton method total n_iter = {}'.format(sum(np.array(n_iter_list2))))
            print('Fixed point iter method avg n_iter = {}'.format(np.mean(np.array(n_iter_list1))))
            print('Newton method avg n_iter = {}'.format(np.mean(np.array(n_iter_list2))))
            print('-------------------------------------------------')
            
        fig, ax = plt.subplots(1,2,figsize=(18,6), constrained_layout=True) 
        
        for tolerance, i in zip(tol_list, [0, 1]):
            ax[i].hist(list2[i], label='Newton', alpha=0.5, color='blue')
            ax[i].hist(list1[i], label='Fixed Point', alpha=0.5, color='red')
            ax[i].grid(), ax[i].legend()
            ax[i].set_title('Tolerance = {}'.format(tolerance))
        fig.suptitle('Histogram of n_iteration at each time steps for different tolerances and methods')
        plt.show() 

        
# ====================================== QUESTION 4 ======================================

def q4():
    """
    Function that plot region of absolute stabity for 
    explicit, implicit and PECE methods
    """
    
    # Setting linspace for plotting
    s = np.linspace(0,2*cmath.pi,1000)
    unit = np.exp(1j*s)
    
    # Computing real and imaginary part of h(s)
    a = 0  # alpha_2
    h = (unit**3+a*unit**2+(-1-a))/(3+2*a)
    re = h.real
    im = h.imag
    
    # Computing real and imaginary part of h(s)
    a = -1  # alpha_2 
    h1 = (unit**3+a*unit**2+(-1-a))/(3+2*a)
    re1 = h1.real
    im1 = h1.imag

    # Plotting
    fig, (ax1,ax2) = plt.subplots(1,2,figsize=(18,10), constrained_layout=True)
    ax1.plot(re, im, color='black')
    ax1.grid()
    ax2.plot(re1, im1, color='black')
    ax2.grid()
    ax1.spines['left'].set_position('zero')
    ax1.spines['right'].set_color('none')
    ax1.spines['bottom'].set_position('zero')
    ax1.spines['top'].set_color('none')
    ax2.spines['left'].set_position('zero')
    ax2.spines['right'].set_color('none')
    ax2.spines['bottom'].set_position('zero')
    ax2.spines['top'].set_color('none')
    ax1.fill(re, im, color='red',alpha=0.2)
    ax1.set_title('Absolute stability region for alpha_2=0', fontsize=18)
    ax2.set_title('Bounndary locus for alpha_2=-1', fontsize=18)
    plt.show()
    
    # Plotting h(s(a)) for alpha approximately in (-3/2,0)
    
    a = np.linspace(-1.2,0,1000)
    t1 = lambda x : 2*np.arctan( np.sqrt((-5-2*np.sqrt(x**2+4))/(2*x-3)) )
    t2 = lambda x : 2*np.arctan( np.sqrt((-5+2*np.sqrt(x**2+4))/(2*x-3)) )

    fig, ax = plt.subplots(1,2, figsize=(18,8), constrained_layout=True)

    for i,t in enumerate([t1,t2]):

        ax[i].plot(a, (np.cos(3*t(a)) + a * np.cos(2*t(a))-1-a)/(2*a+3), color='blue')
        ax[i].grid()
        ax[i].set_title('s_{}(alpha_2)'.format(i+1))
        ax[i].set_xlabel('alpha_2')
        ax[i].set_ylabel('Re(h(alpha_2))')

    plt.show()
    
    
# ====================================== RUN ======================================

print('--------------------------------- QUESTION 1 ---------------------------------')
q1()
print('--------------------------------- QUESTION 2 ---------------------------------')
q2(h=1e-4).compute()
print('--------------------------------- QUESTION 3 ---------------------------------')
q3(h=1e-4).run()
print('--------------------------------- QUESTION 4 ---------------------------------')
q4()