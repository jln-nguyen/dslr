import matplotlib.pyplot as plt
import pandas as pd
import sys

def pair_plot(path: str):
	"""
	"""
	try:
		df = pd.read_csv(path, index_col="Index")
		if df["Hogwarts House"].isna().all():
			raise ValueError("Column 'Hogwarts House' is empty")
		courses = df.select_dtypes(include="number").dropna(axis=1, how="all").columns.tolist()
		num_features = len(courses)
		houses = {"Gryffindor": "darkred", "Hufflepuff": "gold", "Slytherin": "darkgreen", "Ravenclaw": "darkblue"}
		fig, axes = plt.subplots(num_features, num_features, figsize=(20, 20))
		grouped = df.groupby("Hogwarts House")
		for i in range(num_features):
			for j in range(num_features):
				for house, house_df in grouped:
					ax = axes[i, j]
					if (i == j):
						ax.hist(house_df[courses[i]].dropna(), alpha=0.5, label=house)
					else:
						a, b = courses[i], courses[j]
						pair = house_df[[a, b]].dropna()
						ax.scatter(pair[a], pair[b], s=10, alpha=0.5, label=house)
					if j == 0:
						ax.set_ylabel(courses[i], rotation='horizontal', ha='right', fontsize=5)
					if i == num_features - 1:
						ax.set_xlabel(courses[j], fontsize=5)
					ax.set_xticks([])
					ax.set_yticks([])

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
        pair_plot(sys.argv[1])
    
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()