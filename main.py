from models.deep_model import train_deep_model
from models.baseline_model import train_baseline_model
from utils.preprocess import load_data

# Load data
X_train, X_test, y_train, y_test = load_data()

# Train baseline
train_baseline_model(X_train, X_test, y_train, y_test)

# Train deep model
train_deep_model(X_train, X_test, y_train, y_test)
