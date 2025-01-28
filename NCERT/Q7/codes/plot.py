import ctypes
import numpy as np
import matplotlib.pyplot as plt

# Load the shared object file
bernoulli = ctypes.CDLL("./bernoulli.so")

# Set the return type and argument types for the Bernoulli function
bernoulli.bernoulli_random.restype = ctypes.c_int
bernoulli.bernoulli_random.argtypes = [ctypes.c_double]

# Set P(A) = 0.42
p_a = 0.42

# Generate a large number of samples
n_samples = 10000
samples = [bernoulli.bernoulli_random(p_a) for _ in range(n_samples)]

# Calculate probabilities
p_a_hat = sum(samples) / n_samples
p_a_c_hat = 1 - p_a_hat

# Bar plot
plt.figure(figsize=(8, 6))
bars = ['p($A$)', 'p($A^C$)']
probs = [p_a_hat, p_a_c_hat]

plt.bar(bars, probs, color=['blue', 'orange'], alpha=0.7, edgecolor='black')

# Add labels on top of the bars
for i, v in enumerate(probs):
    plt.text(i, v + 0.02, f'{v:.4f}', ha='center', fontsize=12, fontweight='bold')

# Axis labels and title
plt.ylabel("Probability")
plt.ylim(0, 1)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Show the plot
plt.show()

