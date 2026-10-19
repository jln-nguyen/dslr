import pandas as pd
import sys
import numpy as np
import json
from describe import mean, standard_deviation

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def logreg_train(path: str):
    """
    """
    try:
        features = ["Herbology", "Astronomy", "Divination", "Muggle Studies",
            "Ancient Runes", "History of Magic", "Transfiguration",
            "Charms", "Flying"]
        df = pd.read_csv(path, index_col="Index")
        df = df.dropna(subset=["Hogwarts House"])
        features_df = df[features]
        houses_col = df["Hogwarts House"]
        means = {}
        for course in features:
            values = features_df[course].dropna().tolist()
            means[course] = mean(values)
        features_df = features_df.fillna(means)
        stds = {}
        for course in features:
            values = df[course].dropna().tolist()
            stds[course] = standard_deviation(values)
        for course in features:
            features_df[course] = (features_df[course] - means[course]) / stds[course]
            # print(features_df[course])

        X = features_df.to_numpy(dtype=float)
        X = np.hstack((np.ones((X.shape[0], 1)), X))

        houses = ["Gryffindor", "Hufflepuff", "Ravenclaw", "Slytherin"]
        m, n = X.shape
        learning_rate = 0.1
        iterations = 1000
        weights = {}

        for house in houses:
            y = (houses_col == house).to_numpy(dtype=float)
            weight = np.zeros(n)
            for it in range(iterations):
                h = sigmoid(X @ weight)
                gradient = X.T @ (h - y) / m
                weight = weight - learning_rate * gradient
                if it % 100 == 0:
                    cost = -np.mean(y * np.log(h + 1e-15) + (1 - y) * np.log(1 - h + 1e-15))
                    print(house, it, cost)
            weights[house] = weight

        scores = np.column_stack([sigmoid(X @ weights[h]) for h in houses])
        predicted = [houses[i] for i in scores.argmax(axis=1)]
        accuracy = sum(p == t for p, t in zip(predicted, houses_col)) / m
        print(f"Accuracy: {accuracy:.4f}")

        model = {
            "features": features,
            "means": means,
            "stds": stds,
            "weights": {h: weights[h].tolist() for h in houses},
        }
        with open("weights.json", "w") as f:
            json.dump(model, f, indent=4)

    except Exception as e:
        raise RuntimeError(f"{e}")

def main():
    if not len(sys.argv) == 2:
        print("Usage: python describe.py <filename>")
        sys.exit(1)
    
    try:
        if not (sys.argv[1] == "dataset_train.csv" or sys.argv[1] == "dataset/dataset_train.csv"):
            raise ValueError("This program takes 'dataset_train,csv' as paramater")
        logreg_train(sys.argv[1])
    
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
