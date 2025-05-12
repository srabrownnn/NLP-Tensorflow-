from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import accuracy_score
import numpy as np

def train_deep_model(X_train, X_test, y_train, y_test):
    model = Sequential([
        Embedding(input_dim=10000, output_dim=64),
        LSTM(64),
        Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer=Adam(0.001), loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=3, batch_size=128, validation_split=0.2, callbacks=[EarlyStopping(patience=2)])
    
    acc = model.evaluate(X_test, y_test, verbose=0)[1]
    print("Deep Learning Model Accuracy:", acc)
    np.save("models/deep_acc.npy", acc)
