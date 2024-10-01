import pandas as p
import numpy as n
import matplotlib.pyplot as pt

# Define your data
var1 = p.Series([1, 3, 4, 6, 7, 9])
var2 = p.Series([2, 4, 7, 8, 9, 11])

# Create a scatter plot
pt.scatter(var1, var2, color='blue', label='Data Points')

# Perform linear regression using numpy's polyfit
slope, intercept = n.polyfit(var1, var2, 1)

# Add the regression line to the plot
pt.plot(var1, slope*var1 + intercept, color='green', label='Fitted line')

# Add labels and title
pt.xlabel('var1')
pt.ylabel('var2')
pt.title('Scatter plot with linear regression line')

# Add a legend
pt.legend()
# Show the plot
pt.show()
