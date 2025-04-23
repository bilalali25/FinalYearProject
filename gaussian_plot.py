import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Read the data from the CSV file
data = pd.read_excel('../SFE.xlsx')

# Extract the values from the data
values = data['SFE'].values

# Calculate the mean and standard deviation
mu = np.mean(values)
sigma = np.std(values)

# Generate data points for x-axis
x = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)

# Compute the corresponding y-values of the Gaussian distribution
y = (1/(sigma * np.sqrt(2*np.pi))) * np.exp(-(x - mu)**2 / (2*sigma**2))

# Plot the distribution curve
plt.plot(x, y, color='red', label='Distribution Curve')

# Create histogram bars from the data
plt.hist(values, bins=30, density=True, alpha=0.5, label='Histogram')

# Set labels, title, and legend
plt.title('Gaussian Distribution Curve = 3600 atoms')
plt.xlabel('Stacking fault energies (mJ/m$^{2}$)')
plt.ylabel('Count')
plt.legend([f'µ: {mu:.2f}', f'σ: {sigma:.2f}'], loc='upper right')

# Show the plot
plt.show()
