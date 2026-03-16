import pickle
import pandas as pd
from tabulate import tabulate

file_name = input("Enter PKL file name: ")

with open(file_name, "rb") as f:
    data = pickle.load(f)

# convert dictionary to dataframe
df = pd.DataFrame(list(data.items()), columns=["Medicine", "Price"])

print("\nPKL File Content:\n")
print(tabulate(df, headers="keys", tablefmt="grid"))