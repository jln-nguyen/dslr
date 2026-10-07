import matplotlib.pyplot as plt
import pandas as pd
import sys

def mean(values: list) -> float:
    return sum(values) / len(values)


def standard_deviation(values: list) -> float:
    avg = mean(values)
    sum_sq = 0
    for n in values:
        sum_sq += (n - avg) ** 2
    return (sum_sq / (len(values) - 1)) ** 0.5


def covariance(X: list, Y: list) -> float:
    if len(X) != len(Y):
        raise ValueError("X and Y must be the same length")
    x_mean = mean(X)
    y_mean = mean(Y)
    total = 0
    for i in range(len(X)):
        total += (X[i] - x_mean) * (Y[i] - y_mean)
    return total / (len(X) - 1)


def pearson_correlation(X: list, Y: list) -> float:
    return covariance(X, Y) / (standard_deviation(X) * standard_deviation(Y))

def scatter_plot(path: str):
    """
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
    if len(sys.argv) < 2:
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

# import matplotlib.pyplot as plt
# import sys
# import pandas as pd

# def scatter_plot(path: str):
#     """
#     """
#     try:
#         df = pd.read_csv(path, index_col="Index")
#         if df["Hogwarts House"].isna().all():
#             raise ValueError("Column 'Hogwarts House' is empty")
#         courses = df.select_dtypes(include="number").dropna(axis=1, how="all").columns.tolist()

#         for course in courses:
#             col = df[course]
#             standardized = (col - col.mean()) / col.std()
#             plt.scatter(df.index, standardized, s=5, alpha=0.5, label=course)

#         plt.show()
#     except Exception as e:
#         raise RuntimeError(f"{e}")

# def main():
#     if len(sys.argv) < 2:
#         print("Usage: python describe.py <filename>")
#         sys.exit(1)
    
#     try:
#         if not sys.argv[1].endswith(".csv"):
#             raise ValueError("File must be in .csv format")
#         scatter_plot(sys.argv[1])
    
#     except Exception as e:
#         print(f"Error: {e}")
#         sys.exit(1)

# if __name__ == "__main__":
#     main()