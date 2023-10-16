import numpy as np
import matplotlib.pyplot as plt

# Define the function
def f(x):
    return np.log(np.tan(x)**2)

# Generate x values
x = np.linspace(-2*np.pi, 2*np.pi, 1000)

# Calculate corresponding y values
y = f(x)

# Create the plot
plt.figure(figsize=(8, 6))
plt.plot(x, y, label='f(x) = ln(tan^2(x))', color='b')

# Add vertical lines at the discontinuity points
for n in range(-2, 3):
    plt.axvline(n*np.pi, color='r', linestyle='--', label=f'Discontinuity at x = {n}π')

# Set plot properties
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('Graph of f(x) = ln(tan^2(x)) with Discontinuity')
plt.legend()
plt.grid(True)

# Show the plot
plt.show()
