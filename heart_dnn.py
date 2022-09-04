from tensorflow.python.keras.layers import Dense
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential

data = pd.read_csv("heart.csv")
X = data.values[:, :data.shape[1] - 1]
y = data.values[:, data.shape[1] - 1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

models = Sequential()
models.add(Dense(3, activation="relu"))
models.add(Dense(3, activation="relu"))
models.add(Dense(1, activation="sigmoid"))

models.compile(optimizer="adam", loss="binary_crossentropy", metrics=["acc"])
models.fit(X_train, y_train, epochs=150, batch_size=10, validation_data=(X_test, y_test))
print(f"heart csv dnn acc => {models.evaluate(X_test, y_test)}")