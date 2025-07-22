import numpy as np
import matplotlib.pyplot as plt


# epochs at which checkpoints were taken (end of cosine cycle)
checkpoints = [1024, 4096, 8192, 16384, 32768, 65536]

# losses at checkpoint
# all arrays obtained from checkpoint models
l1_0_001_lr_0_0001_training = [0.717237, 0.721894, 0.726413, 0.730826, 0.739504]
l1_0_005_lr_0_0001_training = [0.707593, 0.708883, 0.709994, 0.711421, 0.716785]
l1_0_01_lr_0_0001_training = [0.701676, 0.703094, 0.704196, 0.705433, 0.707593]
l1_0_001_lr_0_0001_test = [0.698249, 0.697406, 0.695617, 0.690668, 0.675739]
l1_0_005_lr_0_0001_test = [0.700964, 0.701882, 0.702189, 0.702371, 0.703206]
l1_0_01_lr_0_0001_test = [0.695826, 0.697804, 0.699005, 0.700076, 0.700827]
l1_0_005_lr_0_01_training = [0.710788, 0.721705, 0.722262, 0.723719]
l1_0_005_lr_0_01_test = [0.702496, 0.696799, 0.696103, 0.695943]

# plot each line into one plot 
# Create figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
fig.suptitle('Model Loss Comparison with Different L1 and Learning Rates', fontsize=16)

# Colors for different L1 values
colors = {
    "0.001": "#1f77b4",  # blue
    "0.005": "#ff7f0e",  # orange
    "0.01": "#2ca02c"    # green
}

# Line styles for different learning rates
linestyles = {
    "0.0001": "-",       # solid
    "0.01": "--"         # dashed
}

# Training Loss Plot
ax1.set_title('Training Loss')
ax1.set_xlabel('Epochs')
ax1.set_ylabel('Loss')
ax1.grid(True, linestyle='--', alpha=0.7)

# Plot each training loss
ax1.plot(checkpoints[:len(l1_0_001_lr_0_0001_training)], l1_0_001_lr_0_0001_training, 
         color=colors["0.001"], linestyle=linestyles["0.0001"], marker='o', 
         label='L1=0.001, LR=0.0001')

ax1.plot(checkpoints[:len(l1_0_005_lr_0_0001_training)], l1_0_005_lr_0_0001_training,
         color=colors["0.005"], linestyle=linestyles["0.0001"], marker='o',
         label='L1=0.005, LR=0.0001')

ax1.plot(checkpoints[:len(l1_0_01_lr_0_0001_training)], l1_0_01_lr_0_0001_training,
         color=colors["0.01"], linestyle=linestyles["0.0001"], marker='o',
         label='L1=0.01, LR=0.0001')

ax1.plot(checkpoints[:len(l1_0_005_lr_0_01_training)], l1_0_005_lr_0_01_training,
         color=colors["0.005"], linestyle=linestyles["0.01"], marker='s',
         label='L1=0.005, LR=0.01')

# Set x-axis to log scale for better visualization
ax1.set_xscale('log')
ax1.legend()

# Test Loss Plot
ax2.set_title('Test Loss')
ax2.set_xlabel('Epochs')
ax2.set_ylabel('Loss')
ax2.grid(True, linestyle='--', alpha=0.7)

# Plot each test loss
ax2.plot(checkpoints[:len(l1_0_001_lr_0_0001_test)], l1_0_001_lr_0_0001_test,
         color=colors["0.001"], linestyle=linestyles["0.0001"], marker='o',
         label='L1=0.001, LR=0.0001')

ax2.plot(checkpoints[:len(l1_0_005_lr_0_0001_test)], l1_0_005_lr_0_0001_test,
         color=colors["0.005"], linestyle=linestyles["0.0001"], marker='o',
         label='L1=0.005, LR=0.0001')

ax2.plot(checkpoints[:len(l1_0_01_lr_0_0001_test)], l1_0_01_lr_0_0001_test,
         color=colors["0.01"], linestyle=linestyles["0.0001"], marker='o',
         label='L1=0.01, LR=0.0001')

ax2.plot(checkpoints[:len(l1_0_005_lr_0_01_test)], l1_0_005_lr_0_01_test,
         color=colors["0.005"], linestyle=linestyles["0.01"], marker='s',
         label='L1=0.005, LR=0.01')

# Set x-axis to log scale for better visualization
ax2.set_xscale('log')
ax2.legend()

# Adjust y-axis limits to better show differences
ax1.set_ylim(0.69, 0.75)
ax2.set_ylim(0.67, 0.71)

plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout to make room for suptitle
plt.show()