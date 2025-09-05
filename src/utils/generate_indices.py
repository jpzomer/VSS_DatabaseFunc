import numpy as np
import json

# Total number of examples
total_examples = 48340

# Validation set size (20%)
val_size = int(0.2 * total_examples)

# Random seed for reproducibility
np.random.seed(42)

# Randomly select 20% of the indices for validation
val_indices = np.random.choice(total_examples, val_size, replace=False)

# Compute the training indices as all indices not in validation
total_indices = np.arange(total_examples)
train_indices = np.setdiff1d(total_indices, val_indices)

# Save both validation and training indices to a JSON file
indices = {
    "train_indices": train_indices.tolist(),
    "val_indices": val_indices.tolist()
}

with open("train_val_indices.json", "w") as f:
    json.dump(indices, f)

print(f"Saved {len(train_indices)} training indices and {len(val_indices)} validation indices to train_val_indices.json.")
