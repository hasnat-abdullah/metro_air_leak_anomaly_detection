import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from pandas.plotting import lag_plot

# Load the dataset
file_path = "../raw_data/oxygen_sample.csv"
data = pd.read_csv(file_path, usecols=['Oxygen', 'time'])

# Convert 'time' column to datetime format
data['time'] = pd.to_datetime(data['time'], errors='coerce')


# Sort data by time
data = data.sort_values('time')

# Display basic statistics for Oxygen
print(data['Oxygen'].describe())

# ---- PLOT 1: Oxygen Levels Over Time ----
plt.figure(figsize=(14, 6))
plt.plot(data['time'], data['Oxygen'], label='Oxygen', alpha=0.7, linewidth=1)
plt.title('Oxygen Levels Over Time')
plt.xlabel('Time')
plt.ylabel('Oxygen Level')
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

# ---- PLOT 2: Boxplot for Anomalies ----
plt.figure(figsize=(10, 6))
sns.boxplot(x=data['Oxygen'])
plt.title('Oxygen Levels - Boxplot (Anomaly Detection)')
plt.grid()
plt.tight_layout()
plt.show()

# ---- PLOT 3: Distribution of Oxygen ----
plt.figure(figsize=(10, 6))
sns.histplot(data['Oxygen'], kde=True, bins=50, color='blue')
plt.title('Distribution of Oxygen Levels')
plt.xlabel('Oxygen Level')
plt.ylabel('Frequency')
plt.grid()
plt.tight_layout()
plt.show()

# ---- PLOT 4: Rolling Average ----
data['Oxygen_RollingMean'] = data['Oxygen'].rolling(window=1000).mean()
plt.figure(figsize=(14, 6))
plt.plot(data['time'], data['Oxygen'], label='Oxygen', alpha=0.3, linewidth=1)
plt.plot(data['time'], data['Oxygen_RollingMean'], label='Rolling Mean (Window=1000)', color='red', linewidth=2)
plt.title('Oxygen Levels with Rolling Mean')
plt.xlabel('Time')
plt.ylabel('Oxygen Level')
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

# ---- PLOT 5: Seasonal Decomposition ----
# Resample data to hourly averages
data_resampled = data.set_index('time').resample('H').mean()

decomposition = seasonal_decompose(data_resampled['Oxygen'].dropna(), model='additive', period=24)
plt.rcParams.update({'figure.figsize': (12, 8)})  # Set the figure size
decomposition.plot()
plt.suptitle('Seasonal Decomposition of Oxygen Levels', fontsize=16)
plt.tight_layout()
plt.show()

# ---- PLOT 6: Heatmap of Hourly Trends ----
data['hour'] = data['time'].dt.hour
data['day'] = data['time'].dt.dayofweek  # Monday=0, Sunday=6
pivot_table = data.pivot_table(index='hour', columns='day', values='Oxygen', aggfunc='mean')
plt.figure(figsize=(10, 6))
sns.heatmap(pivot_table, cmap='coolwarm', annot=False, cbar=True)
plt.title('Average Oxygen Levels by Hour and Day of Week')
plt.xlabel('Day of Week (0=Monday, 6=Sunday)')
plt.ylabel('Hour of Day')
plt.tight_layout()
plt.show()

# ---- PLOT 7: Scatter Plot for Outliers ----
plt.figure(figsize=(14, 6))
plt.scatter(data['time'], data['Oxygen'], alpha=0.5, s=10, c='blue')
plt.title('Scatter Plot of Oxygen Levels')
plt.xlabel('Time')
plt.ylabel('Oxygen Level')
plt.grid()
plt.tight_layout()
plt.show()

# ---- PLOT 8: Heatmap of Hourly Aggregated Data ----
data['hour'] = data.index.hour
data['day'] = data.index.day
hourly_pivot = data.pivot_table(index='day', columns='hour', values='Oxygen', aggfunc='mean')

plt.figure(figsize=(12, 8))
sns.heatmap(hourly_pivot, cmap='coolwarm', annot=False, cbar=True)
plt.title('Heatmap of Hourly Average Oxygen Levels')
plt.xlabel('Hour of Day')
plt.ylabel('Day of Month')
plt.tight_layout()
plt.show()

# ---- PLOT 9: Anomaly Highlight ----
anomalies = data[(data['Oxygen'] < 10) | (data['Oxygen'] > 90)]
plt.figure(figsize=(14, 6))
plt.plot(data['time'], data['Oxygen'], label='Oxygen', alpha=0.7, linewidth=1)
plt.scatter(anomalies['time'], anomalies['Oxygen'], color='red', label='Anomalies', zorder=5)
plt.title('Oxygen Levels with Anomalies Highlighted')
plt.xlabel('Time')
plt.ylabel('Oxygen Level')
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

# ---- PLOT 10: Distribution of Oxygen Levels ----
plt.figure(figsize=(14, 6))
sns.histplot(data['Oxygen'].dropna(), kde=True, bins=50, color='blue', stat='density')
plt.title('Distribution of Oxygen Levels')
plt.xlabel('Oxygen Level')
plt.ylabel('Density')
plt.grid()
plt.tight_layout()
plt.show()

# ---- PLOT 11: Lag Plot ----
plt.figure(figsize=(8, 8))
lag_plot(data['Oxygen'])
plt.title('Lag Plot of Oxygen Levels')
plt.xlabel('Oxygen(t)')
plt.ylabel('Oxygen(t+1)')
plt.grid()
plt.tight_layout()
plt.show()

# ---- PLOT 12: Violin Plot by Hour ----
plt.figure(figsize=(12, 6))
sns.violinplot(x=data['hour'], y=data['Oxygen'])
plt.title('Violin Plot of Oxygen Levels by Hour')
plt.xlabel('Hour of Day')
plt.ylabel('Oxygen Level')
plt.grid()
plt.tight_layout()
plt.show()


# ---- PLOT 13: Scatter Plot with Moving Averages ----
data['short_mavg'] = data['Oxygen'].rolling(window=50).mean()
data['long_mavg'] = data['Oxygen'].rolling(window=200).mean()

plt.figure(figsize=(14, 6))
plt.scatter(data.index, data['Oxygen'], alpha=0.5, label='Oxygen')
plt.plot(data.index, data['short_mavg'], label='Short-Term Avg (50)', color='blue', linewidth=2)
plt.plot(data.index, data['long_mavg'], label='Long-Term Avg (200)', color='orange', linewidth=2)
plt.title('Moving Averages with Scatter Plot')
plt.xlabel('Time')
plt.ylabel('Oxygen Level')
plt.legend(loc='best')
plt.grid()
plt.tight_layout()
plt.show()
