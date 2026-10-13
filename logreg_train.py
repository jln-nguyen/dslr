import pandas as pd
import sys
from describe import mean, standard_deviation

def logreg_train(path: str):
    """
    """
    try:
        features = ["Herbology", "Astronomy", "Divination", "Muggle Studies",
            "Ancient Runes", "History of Magic", "Transfiguration",
            "Charms", "Flying"]
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
    except Exception as e:
        raise RuntimeError(f"{e}")

def main():
    if len(sys.argv) < 2:
        print("Usage: python describe.py <filename>")
        sys.exit(1)
    
    try:
        if not sys.argv[1] == "dataset_train.csv":
            raise ValueError("This program takes 'dataset_train,csv' as paramater")
        logreg_train(sys.argv[1])
    
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
