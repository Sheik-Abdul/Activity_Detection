import pandas as pd
import numpy as np
import random
import flwr as fl

# preprocessing
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
import tensorly as tl
from tensorly.decomposition import parafac

# cnn and lstm
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Bidirectional, Flatten, Dense, Dropout, LSTM
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import KFold

# evaluation
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, f1_score, roc_auc_score

# Load dataset

df = pd.read_csv("C:/Users/9400f/Desktop/CE903-SU_team6/dataset/user04v2.csv")

# Preprocessing
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

scaler = StandardScaler()
X = scaler.fit_transform(X)

rank = 15
tensor = tl.tensor(X)  # Convert the training data to a tensor
factors = parafac(tensor, rank=rank)  # Perform CP decomposition

# Transform the training and testing data using the decomposed factors
X = tl.cp_to_tensor(factors)

encoder = OneHotEncoder(categories='auto')
y = encoder.fit_transform(y.reshape(-1, 1)).toarray()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Reshape the input data to match the expected input shape of the LSTM
X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))


# Define LSTM model
def create_lstm_model(units=128, dropout_rate=0.5, activation='tanh', optimizer='adam',
                      recurrent_activation='tanh', bias_initializer='zeros',
                      kernel_initializer='orthogonal', return_sequences=True,
                      stateful=False, batch_size=32, recurrent_initializer='glorot_uniform'):
    model = Sequential()
    model.add(Bidirectional(LSTM(units=units, return_sequences=True, recurrent_activation=recurrent_activation,
                                 bias_initializer=bias_initializer, kernel_initializer=kernel_initializer,
                                 recurrent_initializer=recurrent_initializer),
                            input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Dropout(dropout_rate))
    model.add(Bidirectional(LSTM(units=units, return_sequences=True, recurrent_activation=recurrent_activation,
                                 bias_initializer=bias_initializer, kernel_initializer=kernel_initializer,
                                recurrent_initializer=recurrent_initializer)))
    model.add(Dropout(dropout_rate))
    model.add(Bidirectional(LSTM(units=units, recurrent_activation=recurrent_activation,
                                 bias_initializer=bias_initializer, kernel_initializer=kernel_initializer,
                                recurrent_initializer=recurrent_initializer)))
    model.add(Dropout(dropout_rate))
    model.add(Dense(units=y.shape[1], activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model


model = KerasClassifier(build_fn=create_lstm_model, epochs=100, batch_size=32, verbose=0)
early_stopping = EarlyStopping(monitor='val_loss', patience=3)

model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test), callbacks=early_stopping)

y_pred = model.predict_proba(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y_test, axis=1)

print("Evaluation Metrics:")
accuracy = accuracy_score(y_true_classes, y_pred_classes)
print("Accuracy:", accuracy)
precision = precision_score(y_true_classes, y_pred_classes, average='weighted', zero_division=1)
print("Precision:", precision)
recall = recall_score(y_true_classes, y_pred_classes, average='weighted')
print("Recall:", recall)
f1 = f1_score(y_true_classes, y_pred_classes, average='weighted')
print("F1 Score:", f1)
roc_auc = roc_auc_score(y_test, y_pred, average='weighted', multi_class='ovr')
print("ROC AUC:", roc_auc)


# Define flower client and perform evaluation
class FlowerClient(fl.client.NumPyClient):
    def __init__(self, model):
        self.model = model
        self.x_train = X_train
        self.y_train = y_train
        self.x_test = X_test
        self.y_test = y_test

    def get_parameters(self, config):
        return [val.numpy() for val in self.model.model.weights]

    def set_parameters(self, parameters):
        self.model.model.set_weights(parameters)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        epochs = config.get("epochs", 1)  # Use default value 1 if "epochs" is not set
        batch_size = config.get("batch_size", 32)  # Use default value 32 if "batch_size" is not set
        history = self.model.model.fit(self.x_train, self.y_train, epochs=epochs, batch_size=batch_size, verbose=0)
        return [val.numpy() for val in self.model.model.weights], len(self.x_train), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, f1 = self.model.model.evaluate(self.x_test, self.y_test, verbose=0)
        print("F1 score results before FedAvg:", f1)
        return loss, len(self.x_test), {"F1 Score": f1}


# Start Flower client
fl.client.start_numpy_client(
    server_address="127.0.0.1:8080",
    client=FlowerClient(model),
    grpc_max_message_length=1024 * 1024 * 1024
)
