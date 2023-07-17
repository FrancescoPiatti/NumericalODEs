import numpy as np 
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

# ---------------------------------------------------------------------------------------------------------------- #
# ---------------------------------------------------------------------------------------------------------------- #

class exercise_1:
    
    def __init__(self):
        """
        Initializing default attributes
        """
        
        self.t0 = 0
        self.tN = 10
        self.x0 = 100
        
    @staticmethod
    def f(t, x):
        return (5 - x - 5025 * np.exp(-8*t)) / 8
    
    @staticmethod
    def exact_solution(t):
        return 320 * np.exp(-t/8) / 21+ 1675 * np.exp(-8*t) / 21 + 5
    
    def euler_method(self, h):
        """
        Input : h - float > 0 : step size
        Output : x - numpy.array : Values of numerical solution for each t = t + ih, i from 0 to int(t_N/h)
        """
        steps = int((self.tN-self.t0) / h)
        t_list = np.linspace(self.t0, self.tN, steps+1)
        x_list = [self.x0]
                
        for t in t_list[:-1]:
            x_nplus1 = x_list[-1] + h * self.f(t, x_list[-1])
            x_list.append(x_nplus1)

        return np.array(x_list)
    
    def results(self, h_list):
        """
        Input : h - list of floats > 0 : step sizes 
        Outputs :
           - Plot of numerical solution for different step sizes and exact solution
           - Table of global errors at t = 10 for each of step size analysed
        """
        
        print('___________________________________________________________________________', '\n')
        print('EXERCISE 1')
        
        plt.figure(figsize=(18,10))
        plt.title('Numerical solutions for exercise 1', fontsize=18)
        global_error_list = []
        exact = exercise_1.exact_solution(10)       # Computing exact solution at time=10
        
        for h in h_list:
            
            label1 = 'h = {}'.format(h)
            numerical_sol = self.euler_method(h)    # Compute numerical solution
            plt.plot(np.linspace(self.t0, self.tN, len(numerical_sol)), numerical_sol, label=label1)
            
            # Compute global error at time 10
            global_error_list.append(np.abs(exact-numerical_sol[-1]))
        
        # Plot exact solution
        t_range = np.linspace(self.t0, self.tN, 1000)
        plt.plot(t_range, exercise_1.exact_solution(t_range), 'r--', label='Exact')
            
        plt.grid()
        plt.legend(fontsize=14)
        plt.show()
        
        # Print errors at time 10 for different h
        print('------------------------------------------------------------')
        print('Global error for forward Euler method at t=10')
        print('------------------------------------------------------------')
        print('Exact solution at t=10 = {} \n'.format(np.round(exact,7)))
        for i,h in enumerate(h_list):
            print('h = {0:5},   error = {1:10},  error/h = {2}'.format(h,np.round(global_error_list[i],7), 
                                                                   np.round(global_error_list[i]/h,7)))
            
        print('------------------------------------------------------------')           
        
# ---------------------------------------------------------------------------------------------------------------- #
# ---------------------------------------------------------------------------------------------------------------- #

class exercise_2:
    
    def __init__(self):
        """
        Initializing default attributes
        """
        
        self.t0 = 0
        self.tN = 3
        self.x0 = 1
        self.x0_prime = 0
        
    @staticmethod    
    def f(t, u, v):
        u_prime = v
        v_prime = t**2 - 3*v - 2*u
        
        return np.array([u_prime, v_prime])
    
    @staticmethod
    def f_prime(t, u, v):
        uv_prime = exercise_2.f(t, u, v)
        u_second = uv_prime[1]
        v_second = 2*t - 3 * uv_prime[1] - 2 * uv_prime[0]
        
        return np.array([u_second, v_second])
    
    @staticmethod
    def f_second(t, u, v):
        uv_second = exercise_2.f_prime(t, u, v)
        u_third = uv_second[1]
        v_third = 2 - 3 * uv_second[1] - 2 * uv_second[0]
        
        return np.array([u_third, v_third])
    
    @staticmethod
    def exact_solution(t):
        return (2*t**2-6*t-3*np.exp(-2*t)+7)/4
        
    
    def ts3_method(self, h, return_global_error=False):
        """
        Input : h - float > 0 : step size
        Output : x - numpy.array : Values of numerical solution for each t = t + ih, i from 0 to int(t_N/h)
        """
        
        steps = int((self.tN-self.t0) / h )
        sol_matrix = np.zeros((2, steps+1)) 
        sol_matrix[:,0] = np.array([self.x0, self.x0_prime])
        t_list = np.linspace(self.t0, self.tN, steps+1)
        
        for i,t in enumerate(t_list[:-1]): 
            
            f_n = self.f(t, sol_matrix[0, i], sol_matrix[1, i])
            f_n_prime = self.f_prime(t, sol_matrix[0, i], sol_matrix[1, i])
            f_n_second = self.f_second(t, sol_matrix[0, i], sol_matrix[1, i])
            x_nplus1 = sol_matrix[:, i] + h * f_n + h**2 * f_n_prime / 2 + h**3 * f_n_second / 6
            sol_matrix[:,i+1] = x_nplus1
        
        if return_global_error is False:
            return np.squeeze(sol_matrix[0,:])
        else:
            
            exact = self.exact_solution(3)
            global_error = np.abs(exact-sol_matrix[0,-1])
            return global_error
            
    
    def results(self, h_list, n_steps_list=np.arange(4,20,1)):
        """
        Input : h - list of floats > 0 : step sizes 
        Outputs :
           - Plot of numerical solution for different step sizes and exact solution
           - Table of global errors at t = 3 for each of step size analysed
           - Plot of global errors at t = 3 against step sizes analysed
        """
        
        print('___________________________________________________________________________', '\n')
        print('EXERCISE 2')
        plt.figure(figsize=(18,10))
        plt.title('Numerical solutions for exercise 2', fontsize=18)
        exact = exercise_2.exact_solution(3)
        
        for h in h_list:
            
            label1 = 'h = {}'.format(h)
            numerical_sol = self.ts3_method(h)
            plt.plot(np.linspace(self.t0, self.tN, len(numerical_sol)), numerical_sol, label=label1)
            
        t_range = np.linspace(self.t0, self.tN, 1000)
        plt.plot(t_range, exercise_2.exact_solution(t_range), 'r--', label='Exact')      # Plot exact solution
        plt.grid()
        plt.legend()
        plt.show()
        
        
        # Plotting global error at time t=3 for against number of steps
        plt.figure(figsize=(18,10))
        plt.title('Global error for integer number of steps at time = 3', fontsize=18)
        global_error_list = []
        
        # Computing global error
        for steps in n_steps_list:
            h = (self.tN-self.t0) / steps
            global_error_list.append(self.ts3_method(h, return_global_error=True))
        plt.plot(n_steps_list,global_error_list, color='blue')
        plt.axhline(y = 1e-3, color = 'black', linestyle = '-')
        plt.scatter(n_steps_list,global_error_list, color='red')
        plt.grid()
        plt.show()
        
        # Print errors at time 3 for different h
        print('------------------------------------------------------------')
        print('Global error for TS(3) method at t=3')
        print('------------------------------------------------------------')
        print('Exact solution at t=3 = {} \n'.format(np.round(exact,7)))
        for steps,e in zip(n_steps_list[:8],global_error_list[:8]):
            print('steps = {0:3},   error = {1:10}'.format(steps,np.round(e,7)))
            
        print('------------------------------------------------------------')     

# ---------------------------------------------------------------------------------------------------------------- #
# ---------------------------------------------------------------------------------------------------------------- #
        
class exercise_4:
    
    def __init__(self):
        """
        Initializing default attributes
        """
        
        self.sigma = 10
        self.rho = 28
        self.beta = 8/3
        self.x0, self.y0, self.z0 = 1, 1, 1
        self.t0, self.tN = 0, 100
        
    
    def f(self, t, x, y, z):
        
        x_prime = self.sigma * (y - x)
        y_prime = x * (self.rho - z) - y
        z_prime = x * y - self.beta * z
        
        return np.array([x_prime, y_prime, z_prime])
    
    
    def numerical_solution(self, h):
        """
        Input : h - float > 0 : step size
        Output : x - numpy.array : Values of numerical solution for each t = t + ih, i from 0 to int(t_N/h)
        """
        
        steps = int((self.tN-self.t0) // h + 1)
        sol_matrix = np.zeros((3, steps+1))
        sol_matrix[:,0] = np.array([self.x0, self.y0, self.z0])
        t = self.t0 

        sol_matrix[:,1] = sol_matrix[:,0] + h * self.f(t, sol_matrix[0, 0], sol_matrix[1, 0], sol_matrix[2, 0])
        t += h
        sol_matrix[:,2] = sol_matrix[:,1] + h * self.f(t, sol_matrix[0, 1], sol_matrix[1, 1], sol_matrix[2, 1])
        t += h
        
        for i in range(2,steps):
            
            f_n = self.f(t, sol_matrix[0, i], sol_matrix[1, i], sol_matrix[2, i])
            f_nm1 = self.f(t-h, sol_matrix[0, i-1], sol_matrix[1, i-1], sol_matrix[2, i-1])
            f_nm2 = self.f(t-2*h, sol_matrix[0, i-2], sol_matrix[1, i-2], sol_matrix[2, i-2])
            x_nplus1 = sol_matrix[:, i] + h * (23 * f_n - 16 * f_nm1 + 5 * f_nm2) / 12
            sol_matrix[:,i+1] = x_nplus1
            t += h
        
        return sol_matrix    
    
    def plot(self, h):
        
        print('___________________________________________________________________________', '\n')
        print('EXERCISE 4')
        
        # syntax for 3-D projection
        fig = plt.figure(figsize=(18,10))
        ax = plt.axes(projection ='3d')
        
        solution = self.numerical_solution(h)
        
        # plotting
        ax.plot3D(solution[0,:], solution[1,:], solution[2,:], 'blue', alpha=0.7)
        ax.set_title('3D line plot Lorentz attractor', fontsize=18)
        plt.show()
        
        
# ---------------------------------------------------------------------------------------------------------------- #
# ---------------------------------------------------------------------------------------------------------------- #


# Run results
exercise_1().results(h_list=[0.1,0.05,0.025])
exercise_2().results(h_list=[0.1,0.05,0.025])
exercise_4().plot(h=0.01)