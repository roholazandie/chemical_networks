import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# Adjusted Constants
k1 = 1.34
k2 = 1.6e-3
k3 = 1.6e-3
k4 = 4.5
k5 = 1e-2
f = 0.5

# Initial concentrations
A = 1.0
B = 1.0
X0 = 0.001
Y0 = 0.001
Z0 = 0.001
initial_conditions = [X0, Y0, Z0]

# Time points
t = np.linspace(0, 500, 10000)  # Extend time and increase resolution

# Differential equations
def oregonator(y, t, A, B, k1, k2, k3, k4, k5, f):
    X, Y, Z = y
    dX_dt = k1 * A * Y - k2 * X * Y + k3 * B * X - 2 * k4 * X**2
    dY_dt = -k1 * A * Y - k2 * X * Y + f * k5 * Z
    dZ_dt = k3 * B * X - k5 * Z
    return [dX_dt, dY_dt, dZ_dt]

# Solve ODEs
sol = odeint(oregonator, initial_conditions, t, args=(A, B, k1, k2, k3, k4, k5, f))

# Extract solutions
X = sol[:, 0]
Y = sol[:, 1]
Z = sol[:, 2]

# Plot the results
plt.figure(figsize=(12, 8))
plt.plot(t, X, label='X', color='r')
plt.plot(t, Y, label='Y', color='b')
plt.plot(t, Z, label='Z', color='g')
plt.xlabel('Time')
plt.ylabel('Concentration')
plt.title('Belousov-Zhabotinsky Reaction (Oregonator Model)')
plt.legend()
plt.show()

# Create a DataFrame
data = pd.DataFrame({'Time': t, 'X': X, 'Y': Y, 'Z': Z})

# import ace_tools as tools; tools.display_dataframe_to_user(name="BZ Reaction Data", dataframe=data)
