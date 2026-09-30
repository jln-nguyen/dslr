import sys
import pandas as pd

def truncate(text: str, width: int) -> str:
    """"""
    if len(text) > width:
        return text[:width - 3] + "..."
    return text

def print_describe(data: list, stats: dict):
    """
    """
    label_w = 8
    col_w = 16
    stats_names = list(stats[data[0]].keys())

    print(" " * label_w, end="")
    for col in data:
        print(f"{truncate(col, col_w - 1):<{col_w}}", end="")
    print()
    for stat in stats_names:
        print(f"{stat:<{label_w}}", end="")
        for col in data:
            print(f"{stats[col][stat]:<{col_w}.6f}", end="")
        print()

def calculate_quartiles(data: list, q: float) -> float:
    """
    """
    pos = (len(data) - 1) * q
    lower = int(pos)
    frac = pos - lower
    if lower + 1 >= len(data):
        return data[lower]
    return data[lower] + (data[lower + 1] - data[lower]) * frac

def ft_describe(path: str):
    """
    """
    try:
        df = pd.read_csv(path, index_col="Index")
        # print (df)
        numerical_df = df.select_dtypes(include='number')
        numerical_df = numerical_df.dropna(axis=1, how='all')
        numerical_cols = numerical_df.columns.tolist()
        result = {}
        for col in numerical_cols:
            total = 0
            sorted_col = []
            clean_col = df[col].dropna()
            for n in clean_col:
                sorted_col.append(n)
                total = total + n
            sorted_col.sort()
            sum_sq = 0
            mean = total/len(sorted_col)
            for n in clean_col:
                sum_sq += (n - mean) ** 2
            std = (sum_sq / (len(sorted_col) - 1)) ** 0.5
            result[col] = {
                "count": len(sorted_col),
                "mean": mean,
                "std" : std,
                "min": sorted_col[0],
                "25%": calculate_quartiles(sorted_col, 0.25),
                "50%": calculate_quartiles(sorted_col, 0.5),
                "75%": calculate_quartiles(sorted_col, 0.75),
                "max": sorted_col[-1]
            }
            # print(clean_col.describe())
        print_describe(numerical_cols, result)
    except Exception as e:
        raise RuntimeError(f"{e}")

def main():
    if len(sys.argv) < 2:
        print("Usage: python describe.py <filename>")
        sys.exit(1)
    
    try:
        if not sys.argv[1].endswith(".csv"):
            raise ValueError("File must be in .csv format")
        ft_describe(sys.argv[1])
    
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()