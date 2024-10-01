# -*- coding: utf-8 -*-
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

# Data
datasets = ['Abalone', 'Adult', 'Hcvdat0', 'Obesity', 'Room', 'Wholesale']
fdpc_values = {
    'Abalone': [0.9607, 0.9347, 0.8056, 0.7767, 0.9164],
    'Adult': [0.9859, 0.9765, 0.9697, 0.9562, 0.9408],
    'Hcvdat0': [0.9016, 0.9025, 0.9028, 0.9027, 0.9022],
    'Obesity': [0.9737, 0.9476, 0.9346, 0.9238, 0.8290],
    'Room': [0.8180, 0.8194, 0.8164, 0.8161, 0.8093],
    'Wholesale': [0.9436, 0.9606, 0.9242, 0.9242, 0.9038]
}
dpc_values = {
    'Abalone': 0.2112,
    'Adult': 0.7172,
    'Hcvdat0': 0.4751,
    'Obesity': 0.3819,
    'Room': 0.2119,
    'Wholesale': 0.5878
}
alphas = [0.02, 0.04, 0.06, 0.08, 0.10]

# Plot settings
fig, axs = plt.subplots(2, 3, figsize=(18, 12))
fig.subplots_adjust(hspace=0.3, wspace=0.3)

# Plot each dataset
for i, dataset in enumerate(datasets):
    row, col = divmod(i, 3)
    ax = axs[row, col]

    # Plot FDPC values
    ax.plot(alphas, fdpc_values[dataset], marker='o', label='FDPC')

    # Plot DPC horizontal line
    ax.axhline(y=dpc_values[dataset], color='r', linestyle='--', label='DPC')

    # Labels and title
    ax.set_title(dataset)
    ax.set_xlabel('Alpha')
    ax.set_ylabel('Balance')
    ax.legend()

plt.suptitle('Balance Comparison between FDPC and DPC')
plt.savefig('combined_plot.png')
plt.show()