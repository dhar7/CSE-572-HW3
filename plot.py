import matplotlib.pyplot as plt
import numpy as np

# --- Your Data ---
ks = [5, 10, 20, 30, 40, 50, 60]
user_based_rmse = [1.0651, 1.0258, 1.0094, 1.0056, 1.0039, 1.0033, 1.0033]
item_based_rmse = [1.2449, 1.1923, 1.1522, 1.1296, 1.1100, 1.0932, 1.0786]
# -------------------

# Create the plot
plt.figure(figsize=(10, 6))

# Plot User-Based CF
plt.plot(ks, user_based_rmse, marker='o', linestyle='-', label='User-Based CF (RMSE)', color='royalblue')

# Plot Item-Based CF
plt.plot(ks, item_based_rmse, marker='s', linestyle='-', label='Item-Based CF (RMSE)', color='darkorange')

# --- Add Labels and Title ---
plt.title('Impact of K Neighbors on RMSE (MSD Sim)', fontsize=16)
plt.xlabel('Number of Neighbors (K)', fontsize=12)
plt.ylabel('Average RMSE', fontsize=12)
plt.legend(fontsize=11)

# Add grid lines for readability
plt.grid(True, linestyle='--', alpha=0.7)

# Set the x-axis ticks to match your data points
plt.xticks(ks)

# Invert y-axis so "better" (lower RMSE) is up (optional, but common in some fields)
# plt.gca().invert_yaxis() 
# We will keep the standard (lower is better)

# Save the plot as a file
plt.savefig('k_neighbors_impact_plot.png')

# Display the plot
print("Plot saved as 'k_neighbors_impact_plot.png'. Displaying plot...")
plt.show()