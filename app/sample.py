import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Define the DataFrame 'df' based on the provided data
data = {
    'Year': [1959, 1958, 1957],
    'Mean cumulative mass balance': [-1.431, -0.963, -0.094],
    'Number of observations': [13.0, 12.0, 12.0]
}
df = pd.DataFrame(data)

# Plot the distribution of the "Mean cumulative mass balance" column
plt.figure(figsize=(8, 5))
sns.histplot(df['Mean cumulative mass balance'], bins=10, kde=True)
plt.title('Distribution of Mean Cumulative Mass Balance')
plt.xlabel('Mean Cumulative Mass Balance')
plt.ylabel('Frequency')
plt.show()
