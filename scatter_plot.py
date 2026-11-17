import matplotlib.pyplot as plt
import pandas as pd
import sys
from describe import mean, standard_deviation

def covariance(X: list, Y: list) -> float:
    """
    Calculates the covariance between two lists of numbers.
    """
    if len(X) != len(Y):
        raise ValueError("X and Y must be the same length")
    x_mean = mean(X)
    y_mean = mean(Y)
    total = 0
    for i in range(len(X)):
        total += (X[i] - x_mean) * (Y[i] - y_mean)
    return total / (len(X) - 1)


def pearson_correlation(X: list, Y: list) -> float:
    """
    Calculates the Pearson correlation coefficient between two lists of numbers.
    """
    return covariance(X, Y) / (standard_deviation(X) * standard_deviation(Y))

def scatter_plot(path: str):
    """
    Takes a .csv dataset as a parameter and displays a scatter plot of the two most correlated numerical features in the dataset.
    """
    try:
        df = pd.read_csv(path, index_col="Index")
        courses = df.select_dtypes(include="number").dropna(axis="columns", how="all").columns.tolist()
        best_pair = None
        best_r = 0
        for i in range(len(courses)):
            for j in range(i + 1, len(courses)):
                a, b = courses[i], courses[j]
                pair = df[[a, b]].dropna()
                X = pair[a].tolist()
                Y = pair[b].tolist()
                # print(a, b, pearson_correlation(X, Y))
                r = pearson_correlation(X, Y)
                if abs(r) > abs(best_r):
                    best_r = r
                    best_pair = (a, b)
        print(f"Most similar: {best_pair[0]} / {best_pair[1]} (r = {best_r:.4f})")
        a, b = best_pair
        pair = df[[a, b]].dropna()
        plt.scatter(pair[a], pair[b], s=10, alpha=0.5)
        plt.xlabel(a)
        plt.ylabel(b)
        plt.title(f"{a} vs {b}  (r = {best_r:.3f})")
        plt.show()

    except Exception as e:
        raise RuntimeError(f"{e}")

def main():
    if not len(sys.argv) == 2:
        print("Usage: python describe.py <filename>")
        sys.exit(1)
    
    try:
        if not sys.argv[1].endswith(".csv"):
            raise ValueError("File must be in .csv format")
        scatter_plot(sys.argv[1])
    
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
