# import libraries and packages
# import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# reading the CSV file
df = pd.read_csv('gene_expression.csv')

# displaying the DataFrame
print(df)

# creating a basic histogram
df.hist(by='Cancer Present', figsize=[12, 8], bins=15)
sns.set()
plt.show()
