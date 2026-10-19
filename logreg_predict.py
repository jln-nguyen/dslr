import sys
import json
import pandas as pd
import numpy as np
from logreg_train import sigmoid

def logreg_predict(csv_file: str, weight_file: str):
    """
    """
    try:
        with open(weight_file, "r") as file:
            weight = json.load(file)
        # print(weight)
        features = weight["features"]
        means = weight["means"]
        stds = weight["stds"]
        house_names = list(weight["weights"].keys())
        df = pd.read_csv(csv_file)
        features_df = df[features]
        features_df = features_df.fillna(means)
        for course in features:
            features_df[course] = (features_df[course] - means[course]) / stds[course]
        X = features_df.to_numpy(dtype=float)
        X = np.hstack((np.ones((X.shape[0], 1)), X))
        scores = np.column_stack(
            [sigmoid(X @ np.array(weight["weights"][h])) for h in house_names]
        )
        predicted = [house_names[i] for i in scores.argmax(axis=1)]
        result = pd.DataFrame({"Hogwarts House": predicted}, index=df.index)
        result.index.name = "Index"
        result.to_csv("houses.csv")
    except Exception as e:
        raise RuntimeError(f"{e}")

def main():
    if not len(sys.argv) == 3:
        print("Usage: python describe.py <dataset/dataset_test.csv> <weight.json>")
        sys.exit(1)
    
    try:
        if not (sys.argv[1] == "dataset_test.csv" or sys.argv[1] == "dataset/dataset_test.csv"):
            raise ValueError("This program takes 'dataset_test.csv' as paramater")
        if not (sys.argv[1] == "weights.json" or sys.argv[2] == "weights.json"):
            raise ValueError("This program takes 'weights.json' as second paramater")
        logreg_predict(sys.argv[1], sys.argv[2])
    
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
