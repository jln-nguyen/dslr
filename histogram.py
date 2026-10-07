import matplotlib.pyplot as plt
import sys
import pandas as pd

def histogram(path: str):
    """
    """
    try:
        df = pd.read_csv(path, index_col="Index")
        if df["Hogwarts House"].isna().all():
            raise ValueError("Column 'Hogwarts House' is empty")
        courses = df.select_dtypes(include="number").dropna(axis=1, how="all").columns.tolist()
        houses = {"Gryffindor": "darkred", "Hufflepuff": "gold", "Slytherin": "darkgreen", "Ravenclaw": "darkblue"}
        rows = 4
        cols = 4
        fig, axes = plt.subplots(rows, cols, figsize=(16, 8))
        grouped = df.groupby("Hogwarts House")
        axes = axes.flatten()
        for i, (ax, course) in enumerate(zip(axes, courses)):
            for house, house_df in grouped:
                ax.hist(house_df[course].dropna(), alpha=0.5, label=house)
            ax.set_title(course)
            if i % cols == 0:
                ax.set_ylabel("Number of students")
            ax.set_xlabel("Score")

        for ax in axes[len(courses):]:
            ax.axis("off")
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower right")
        fig.tight_layout()
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
        histogram(sys.argv[1])
    
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()