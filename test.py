import pandas as pd

mine = pd.read_csv("houses.csv", index_col="Index")
truth = pd.read_csv("dataset/dataset_train.csv", index_col="Index")

print(len(mine), len(truth))
print((mine["Hogwarts House"] == truth["Hogwarts House"]).mean())