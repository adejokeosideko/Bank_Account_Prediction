import pickle
from sklearn.linear_model import LogisticRegression
import pandas as pd

# Load your training data
# data = pd.read_csv("your_training_data.csv")
# X_train = data[...]  # your features
# y_train = data[...]  # your target

model = LogisticRegression()
# model.fit(X_train, y_train)  # uncomment and actually fit your model

with open("LR_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Pickle saved!")