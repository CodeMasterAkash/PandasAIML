import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols

# Example data
data = {
    'value': [23, 21, 19, 22, 20, 30, 32, 29, 31, 33, 28, 27, 26, 29, 30],
    'group': ['A'] * 5 + ['B'] * 5 + ['C'] * 5
}

df = pd.DataFrame(data)

# Fit the model
model = ols('value ~ C(group)', data=df).fit()

# Perform ANOVA
anova_table = sm.stats.anova_lm(model, typ=2)

print(anova_table)

# Interpret results
p_value = anova_table['PR(>F)'].iloc[0]  # Use iloc for position-based indexing
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
