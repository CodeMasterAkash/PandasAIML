# import numpy as np
from scipy import stats

# Example data
group1 = [23, 21, 19, 22, 20]
group2 = [30, 32, 29, 31, 33]
group3 = [28, 27, 26, 29, 30]

# Perform one-way ANOVA
f_statistic, p_value = stats.f_oneway(group1, group2, group3)

print(f"F-statistic: {f_statistic}")
print(f"P-value: {p_value}")

# Interpret results
alpha = 0.05
if p_value < alpha:
    print(
        "Reject the null hypothesis - there are significant differences "
        "between the groups."
    )
else:
    print(
        "Fail to reject the null hypothesis - no significant differences "
        "between the groups."
    )
