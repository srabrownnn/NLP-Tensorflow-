from sklearn.metrics import accuracy_score
import numpy as np

# Dummy A/B test (You can extend this for real-case metrics or test sets)
acc_baseline = np.load("models/baseline_acc.npy")
acc_deep = np.load("models/deep_acc.npy")

print(f"Baseline Model Accuracy: {acc_baseline:.4f}")
print(f"Deep Learning Model Accuracy: {acc_deep:.4f}")

if acc_deep > acc_baseline:
    print("✅ Deep Learning model performs better.")
else:
    print("⚠️ Baseline model performs better or equal.")
