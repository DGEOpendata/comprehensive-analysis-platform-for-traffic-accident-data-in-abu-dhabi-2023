python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
data_path = 'Traffic_Accident_Abu_Dhabi_2023.csv'
data = pd.read_csv(data_path)

# Data Cleaning
# Handle missing values
data = data.dropna()

# Exploratory Data Analysis
# Plotting Heatmap for Accident Hotspots
sns.heatmap(data.groupby(['City', 'Street']).size().unstack(), cmap='Reds')
plt.title('Accident Hotspots in Abu Dhabi - 2023')
plt.xlabel('Streets')
plt.ylabel('Cities')
plt.show()

# Time Series Analysis
# Accident trends by time of day
time_of_day = pd.to_datetime(data['Report Time'], format='%H:%M').dt.hour
accidents_by_hour = time_of_day.value_counts().sort_index()
plt.plot(accidents_by_hour.index, accidents_by_hour.values, marker='o')
plt.title('Accidents by Time of Day')
plt.xlabel('Hour')
plt.ylabel('Number of Accidents')
plt.grid()
plt.show()

# Predictive Analytics Example
# Simple Weather Impact Analysis
weather_impact = data.groupby('Weather')['Number of Accidents'].mean()
plt.bar(weather_impact.index, weather_impact.values, color='blue')
plt.title('Average Accidents by Weather Condition')
plt.xlabel('Weather Condition')
plt.ylabel('Average Number of Accidents')
plt.show()
