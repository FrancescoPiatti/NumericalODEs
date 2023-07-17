import numpy as np
import matplotlib.pyplot as plt
import math

class coursework2:
    
    def __init__(self):
        
        self.t0 = 0
        self.tN = 30
        self.x1_0 = 1
        self.x2_0 = 2
        self.A = np.array([[-1,1],[1,-1000]])
        
    @staticmethod
    def b(t):
        
        return np.array([2*np.sin(t), 1000*(np.cos(t)-np.sin(t))])
        
    def f(self, t, x):

        return self.A @ x + self.b(t)
    
    @staticmethod
    def exact_solution(t):
        
        m1 = 2001002/1998005
        m2 = -1994004/1998005
        k1 = -1004/1998005
        k2 = 1999998/1998005

        x1 = lambda t : (998005-1003*np.sqrt(998005))*np.exp(-t*(np.sqrt(998005)+1001)/2)/1996010 + (998005+1003*np.sqrt(998005))*np.exp(-t*(-np.sqrt(998005)+1001)/2)/1996010+k1 * np.cos(t) + m1 * np.sin(t)
        x2 = lambda t : (998005-998*np.sqrt(998005))*np.exp(-t*(-np.sqrt(998005)+1001)/2)/998005 + (998005+998*np.sqrt(998005))*np.exp(-t*(-np.sqrt(998005)+1001)/2)/998005+k2 * np.cos(t) + m2 * np.sin(t)

        return np.array([x1(t), x2(t)])
    
    def explicit_method(self, h):
        
        steps = int(math.ceil(self.tN / h))
        sol = np.zeros((2, steps)) 
        sol = np.hstack((np.array([self.x1_0, self.x2_0]).reshape(2,1), sol))
        t = self.t0
        
        # Parameters
        alpha_2, alpha_1 = -3/2, 1/2
        beta_2, beta_1, beta_0 = 41/24, -40/24, 11/24
        
        # Two forward Eulers
        sol[:,1] = sol[:,0] + h * self.f(t, sol[:,0])
        t += h
        sol[:,2] = sol[:,1] + h * self.f(t, sol[:,1])
        t += h
        i = 2
        
        while t < self.tN-h:
            
            # Explicit method
            part1 = - alpha_2 * sol[:, i] - alpha_1 * sol[:, i-1]
            part2 = h * (beta_2 * self.f(t, sol[:, i]) + beta_1 * self.f(t-h, sol[:, i-1]) + beta_0 * self.f(t-2*h, sol[:, i-2]))
            sol[:, i+1] = part1 + part2
            
            # Update index and t
            t += h
            i += 1
            
        return sol
        
    
    def implicit_method(self, h):

        steps = int(math.ceil(self.tN / h))
        sol = np.zeros((2, steps)) 
        sol = np.hstack((np.array([self.x1_0, self.x2_0]).reshape(2,1), sol))
        t = self.t0
        
        # Parameters
        alpha_2, alpha_1 = -3/2, 1/2
        beta_3, beta_2, beta_0 = 5/9, 1/24, -7/72
        
        # Two forward Eulers
        sol[:,1] = sol[:,0] + h * self.f(t, sol[:,0])
        t += h
        sol[:,2] = sol[:,1] + h * self.f(t, sol[:,1])
        t += h
        i = 2
        
        while t < self.tN-h:
            
            # Explicit method
            part1 = - alpha_2 * sol[:, i] - alpha_1 * sol[:, i-1]
            part2 = h * (beta_3 * self.b(t+h) + beta_2 * self.f(t, sol[:, i]) + beta_0 * self.f(t-2*h, sol[:, i-2]))
            part3 = np.eye(2) - h * beta_3 * self.A  
            sol[:, i+1] = np.linalg.inv(part3) @ (part1 + part2)
            
            # Update index and t
            t += h
            i += 1
            
        return sol
        
        
    def run(self, h=0.0001, h_list=[0.0005, 0.00001]):
        
        numerical_sol_explicit = self.explicit_method(h)
        t_list = np.arange(self.t0, self.tN+h, h)
        exact_sol = self.exact_solution(t_list)
        
        # Plot explicit scheme
        fig, (ax1,ax2) = plt.subplots(1,2,figsize=(18,8), constrained_layout=True)
        
        ax1.plot(t_list, numerical_sol_explicit[0,:], label='x1(t)')
        ax1.plot(t_list, numerical_sol_explicit[1,:], label='x2(t)')
        ax2.plot(numerical_sol_explicit[0,:], numerical_sol_explicit[1,:], 'b-', label='Num')
        ax2.plot(exact_sol[0,:], exact_sol[1,:], 'r--', label='Exact')
        
        plt.suptitle('Explicit scheme', fontsize=18)
        ax1.set_xlabel('Time', fontsize=14), ax1.set_ylabel('x', fontsize=14)
        ax2.set_xlabel('x1(t)', fontsize=14), ax2.set_ylabel('x2(t)', fontsize=14)
        ax1.legend(fontsize=14), ax2.legend(fontsize=14)
        ax1.grid(), ax2.grid()
        plt.show()
        
        # Plot implicit scheme
        fig, (ax1,ax2) = plt.subplots(1,2,figsize=(18,8), constrained_layout=True)
        numerical_sol_implicit = self.implicit_method(h)
        
        ax1.plot(t_list, numerical_sol_implicit[0,:], label='x1(t)')
        ax1.plot(t_list, numerical_sol_implicit[1,:], label='x2(t)')
        ax2.plot(numerical_sol_implicit[0,:], numerical_sol_implicit[1,:], 'b-', label='Num')
        ax2.plot(exact_sol[0,:], exact_sol[1,:], 'r--', label='Exact')
        
        plt.suptitle('Implicit scheme', fontsize=18)
        ax1.set_xlabel('Time', fontsize=14), ax1.set_ylabel('x', fontsize=14)
        ax2.set_xlabel('x1(t)', fontsize=14), ax2.set_ylabel('x2(t)', fontsize=14)
        ax1.legend(fontsize=14), ax2.legend(fontsize=14)
        ax1.grid(), ax2.grid()
        plt.show()
        
        print('------------------------------------')
        print('Global error analysis')
        print('------------------------------------')
        print('h = 10e-5')
        
        exact_value = exact_sol[:,-1]
        
        print('Exact values = ', exact_value)
        print('Numerical values explicit = ', numerical_sol_explicit[:,-1])
        error = np.sqrt((numerical_sol_explicit[0,-1]-exact_value[0])**2+(numerical_sol_explicit[1,-1]-exact_value[1])**2)
        print('Error explicit = ', error)
        print('Error/h^3 explicit = ', error/h**3)
        print('Numerical values implicit = ', numerical_sol_implicit[:,-1])
        error = np.sqrt((numerical_sol_implicit[0,-1]-exact_value[0])**2+(numerical_sol_implicit[1,-1]-exact_value[1])**2)
        print('Error implicit = ', error)
        print('Error/h^3 implicit = ', error/h**3)
            
        print('------------------------------------')
        
        for h1 in h_list:
            
            print('h = ', h1)
            print('Exact values = ', exact_value)
            numerical_sol_t30 = self.explicit_method(h1)
            print('Numerical values explicit = ', numerical_sol_t30[:,-1])
            error = np.sqrt((numerical_sol_t30[0,-1]-exact_value[0])**2+(numerical_sol_t30[1,-1]-exact_value[1])**2)
            print('Error explicit = ', error)
            print('Error/h^3 explicit = ', error/h1**3)
            numerical_sol_t30_im = self.implicit_method(h1)
            print('Numerical values implicit = ', numerical_sol_t30_im[:,-1])
            error = np.sqrt((numerical_sol_t30_im[0,-1]-exact_value[0])**2+(numerical_sol_t30_im[1,-1]-exact_value[1])**2)
            print('Error implicit = ', error)
            print('Error/h^3 implicit = ', error/h1**3)
            print('------------------------------------')
            
        print('DONE')
        
coursework2().run()