import pandas as p
var1 = p. Series([1, 3, 4, 6, 7, 9])
var2 = p. Series([2, 4, 7, 8, 9, 11])
correlation = var2. corr(var1)
print(correlation)
correlation = var1. corr(var2)
print(correlation)
