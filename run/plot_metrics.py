import matplotlib.pyplot as plt
import json
import os

# Path to the saved metrics file
metrics_path = "./checkpoints/train_ted/train_metrics.json"

# Load metrics
if not os.path.exists(metrics_path):
    raise FileNotFoundError(f"Metrics file not found at {metrics_path}")

with open(metrics_path, "r") as f:
    metrics = json.load(f)

train_losses = metrics["train_losses"]
val_losses = metrics["val_losses"]
train_accuracies = metrics["train_accuracies"]
val_accuracies = metrics["val_accuracies"]
epochs = range(1, len(train_losses) + 1)

# Plot Loss Graph
plt.figure(figsize=(10, 5))
plt.plot(epochs, train_losses, label="Training Loss", marker="o")
plt.plot(epochs, val_losses, label="Validation Loss", marker="o")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training vs Validation Loss")
plt.legend()
plt.grid()
plt.savefig("./checkpoints/train_ted/loss_plot.png")
plt.show()

# Plot Accuracy Graph
plt.figure(figsize=(10, 5))
plt.plot(epochs, train_accuracies, label="Training Accuracy", marker="o")
plt.plot(epochs, val_accuracies, label="Validation Accuracy", marker="o")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("Training vs Validation Accuracy")
plt.legend()
plt.grid()
plt.savefig("./checkpoints/train_ted/accuracy_plot.png")
plt.show()
