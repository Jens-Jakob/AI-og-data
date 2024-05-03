import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

file_path = 'Global temperatur.txt'

# Opgaver 2
df = pd.read_csv(file_path, delim_whitespace=True, skiprows=range(0, 38),
                 names=['Date_Number', 'Year', 'Month', 'Day', 'Day_of_Year', 'Anomaly'])

df['Date'] = pd.to_datetime(df[['Year', 'Month', 'Day']])
# Set Date as the index
df.set_index('Date', inplace=True)

# Opgave 3
print(df.isnull().sum())

# Opgave 4
print(df.describe())

df['Date'] = pd.to_datetime(df[['Year', 'Month', 'Day']])

# Plotting the temperature anomaly over time
plt.figure(figsize=(14, 7))
plt.plot(df['Date'], df['Anomaly'], label='Temperature Anomaly', color='blue', linewidth=1)
plt.title('Temperature Anomaly Over Time')
plt.xlabel('Year')
plt.ylabel('Temperature Anomaly (°C)')
plt.grid(True)
plt.legend()
plt.show()

# Opgave 5
fig, axs = plt.subplots(3, 1, figsize=(12, 9))  # 3 subplots in a column

# Plot 1: Temperature Anomalies over Time
axs[0].plot(df.index, df['Anomaly'], label='Temperature Anomaly', color='blue', linewidth=1)
axs[0].set_title('Temperature Anomaly Over Time')
axs[0].set_xlabel('Year')
axs[0].set_ylabel('Temperature Anomaly (°C)')
axs[0].grid(True)
axs[0].legend()

# Plot 2: Rolling Mean of Temperature Anomalies (10-year window)
df['Rolling_Mean'] = df['Anomaly'].rolling(window=3650).mean()  # 10 years * 365 days
axs[1].plot(df.index, df['Rolling_Mean'], label='10-Year Rolling Mean', color='red', linewidth=1)
axs[1].set_title('10-Year Rolling Mean of Temperature Anomalies')
axs[1].set_xlabel('Year')
axs[1].set_ylabel('Temperature Anomaly (°C)')
axs[1].grid(True)
axs[1].legend()

# Plot 3: Histogram of Temperature Anomalies
axs[2].hist(df['Anomaly'], bins=50, color='green', alpha=0.75)
axs[2].set_title('Histogram of Temperature Anomalies')
axs[2].set_xlabel('Temperature Anomaly (°C)')
axs[2].set_ylabel('Frequency')
axs[2].grid(True)

plt.tight_layout()
plt.show()

#Opgave 6 

monthly_avg_anomaly = df['Anomaly'].groupby([df.index.year, df.index.month]).mean().unstack()
yearly_avg_anomaly = df['Anomaly'].groupby(df.index.year).mean()

# Plotting
fig, ax = plt.subplots(figsize=(15, 9))

# Heatmap
sns.heatmap(monthly_avg_anomaly, cmap='coolwarm', ax=ax, cbar_kws={'label': 'Temperature Anomaly (°C)'})
ax.set_title('Monthly Temperature Anomalies Over the Years')
ax.set_xlabel('Month')
ax.set_ylabel('Year')

ax2 = ax.twinx()
ax2.plot(yearly_avg_anomaly.index, yearly_avg_anomaly, color='yellow', marker='o', label='Yearly Avg Anomaly')
ax2.set_ylabel('Yearly Average Temperature Anomaly (°C)')
ax2.legend(loc='upper right')
plt.show()
