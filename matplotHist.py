# Importing required libraries
import pandas as pd
import matplotlib.pyplot as plt

# Creating a Data frame
values = pd.DataFrame({
    'Length': [2.7, 8.7, 3.4, 2.4, 1.9],
    'Breadth': [4.24, 2.67, 7.6, 7.1, 4.9],
    'Height': [5.8, 5.5, 7.8, 10.88, 0.1],
    'Weight': [20, 40.8, 55.8, 7.2, 48]
})

# Plotting Histogram for the 'Length' column
hist = values['Length'].hist(bins=8)

# Adding title and labels
plt.title('Histogram for Length Column')
plt.xlabel('Length')
plt.ylabel('Frequency')

# Display the histogram
plt.show()
