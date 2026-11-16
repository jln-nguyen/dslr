import json
import sys
import numpy as np
import pandas as pd
from describe import mean, standard_deviation

FEATURES = ["Herbology", "Astronomy", "Divination", "Muggle Studies",
            "Ancient Runes", "History of Magic", "Transfiguration",
            "Charms", "Flying"]
HOUSES = ["Gryffindor", "Hufflepuff", "Ravenclaw", "Slytherin"]

# def split_data(df: pd.DataFrame, train_ratio: float = 0.8, seed: int = 42):
#     train_df = df.sample(frac=train_ratio, random_state=seed)
#     val_df = df.drop(train_df.index)
#     return train_df, val_df

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def load_train_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, index_col="Index")
    return df.dropna(subset=["Hogwarts House"])

def compute_stats(df: pd.DataFrame, features: list):
    means, stds = {}, {}
    for course in features:
        values = df[course].dropna().tolist()
        means[course] = mean(values)
        stds[course] = standard_deviation(values)
    return means, stds

def preprocess(df: pd.DataFrame, features: list, means: dict, stds: dict):
    features_df = df[features].fillna(means)
    for course in features:
        features_df[course] = (features_df[course] - means[course]) / stds[course]
    X = features_df.to_numpy(dtype=float)
    return np.hstack((np.ones((X.shape[0], 1)), X))

# def compute_cost(h, y) -> float:
#     return -np.mean(y * np.log(h + 1e-15) + (1 - y) * np.log(1 - h + 1e-15))

def gradient_descent(X, y, learning_rate: float, iterations: int):
    m, n = X.shape
    weights = np.zeros(n)
    for it in range(iterations):
        h = sigmoid(X @ weights)
        gradient = X.T @ (h - y) / m
        weights = weights - learning_rate * gradient
        # if it % 100 == 0:
        #     print(it, compute_cost(h, y))
    return weights

def train_one_vs_all(X, houses_col, house_names, learning_rate, iterations):
    weights = {}
    for house in house_names:
        # print(f"--- {house} ---")
        y = (houses_col == house).to_numpy(dtype=float)
        weights[house] = gradient_descent(X, y, learning_rate, iterations)
    return weights

def predict_houses(X, weights: dict) -> list:
    names = list(weights.keys())
    scores = np.column_stack([sigmoid(X @ weights[h]) for h in names])
    return [names[i] for i in scores.argmax(axis=1)]

# def accuracy(predicted: list, truth) -> float:
    # good = sum(p == t for p, t in zip(predicted, truth))
    # return good / len(predicted)

def save_model(path: str, features, means, stds, weights):
    model = {
        "features": features,
        "means": means,
        "stds": stds,
        "weights": {h: w.tolist() for h, w in weights.items()},
    }
    with open(path, "w") as f:
        json.dump(model, f, indent=4)

def logreg_train(path: str):
    try:
        df = load_train_data(path)
        means, stds = compute_stats(df, FEATURES)
        X = preprocess(df, FEATURES, means, stds)
        weights = train_one_vs_all(X, df["Hogwarts House"], HOUSES, 0.1, 1000)
        predicted = predict_houses(X, weights)
        # print(f"Accuracy: {accuracy(predicted, df['Hogwarts House']):.4f}")
        save_model("weights.json", FEATURES, means, stds, weights)
        
    except Exception as e:
        raise RuntimeError(f"{e}") from e

# def validate(path: str):
#     df = load_train_data(path)
#     train_df, val_df = split_data(df)
#     print(len(train_df), len(val_df))

#     means, stds = compute_stats(train_df, FEATURES)

#     X_train = preprocess(train_df, FEATURES, means, stds)
#     X_val = preprocess(val_df, FEATURES, means, stds)

#     weights = train_one_vs_all(X_train, train_df["Hogwarts House"],
#                                HOUSES, 0.1, 1000)

#     acc_train = accuracy(predict_houses(X_train, weights), train_df["Hogwarts House"])
#     acc_val = accuracy(predict_houses(X_val, weights), val_df["Hogwarts House"])
#     print(f"Train accuracy: {acc_train:.4f}")
#     print(f"Validation accuracy: {acc_val:.4f}")

def main():
    if not len(sys.argv) == 2:
        print("Usage: python describe.py <filename>")
        sys.exit(1)
    
    try:
        if not (sys.argv[1] == "dataset_train.csv" or sys.argv[1] == "dataset/dataset_train.csv"):
            raise ValueError("This program takes 'dataset_train,csv' as paramater")
        # validate(sys.argv[1])
        logreg_train(sys.argv[1])
    
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
