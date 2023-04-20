import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# Define the van der Pol equation
def van_der_pol(t, y, mu):
    return [y[1], mu * (1 - y[0] ** 2) * y[1] - y[0]]

def solve_plot(f, mu, span, y0, precision = 1e-6, plot = 'solution'):
    sol = solve_ivp(lambda t, y: f(t, y, mu), 
                    span,
                    y0,
                    atol=precision,
                    rtol=precision,
                    method = 'RK45',
                    **{'max_step':0.01})
    t_vals = sol.t
    y_vals = sol.y
    if plot == 'solution':
        plt.plot(t_vals, y_vals[0], label='y(t)')
        plt.plot(t_vals, y_vals[1], label="y'(t)")
        plt.legend()
        plt.xlabel('t')
    if plot == 'phase':
        plt.plot(y_vals[0], y_vals[1])
        plt.xlabel("y")
        plt.ylabel("y'")
    plt.show()

t_span = [0,100]
y0     = [0.01, 0.01]
solve_plot(van_der_pol, 0.5, t_span, y0)
