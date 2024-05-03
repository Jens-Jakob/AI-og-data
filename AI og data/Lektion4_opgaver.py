import pandas as pd 
import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.impute import MissingIndicator
#import label encoder
from sklearn.preprocessing import LabelEncoder


#           Opgave 2

file_path = '/Users/jens-jakobskotingerslev/Desktop/AI og data/recipeData.csv'

# Attempt to read with ISO-8859-1 encoding
df = pd.read_csv(file_path, encoding='ISO-8859-1')

print(df.shape)
plt.figure(figsize=(12, 8))
sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
plt.title('Heatmap of Missing Data in Recipe Dataset')
plt.xlabel('Columns')
plt.ylabel('Data Index')
#plt.xticks(rotation=45)
plt.show()

print(df['BrewMethod'].unique())

# plot for korrelation mellem features med null value features 
indicator = MissingIndicator(features='missing-only', error_on_new=True)

missing_data = indicator.fit_transform(df)

missing_data_df = pd.DataFrame(missing_data, columns=[f'missing_{col}' for col in df.columns[indicator.features_]])

correlation_matrix = missing_data_df.corr()

# Plot for hvordan "brewMethod" hænger sammen med null value features 
missing_by_brewmethod = df.groupby('BrewMethod').apply(lambda x: x.isnull().mean()).loc[:, ['PrimaryTemp', 'PitchRate', 'MashThickness','PrimingMethod','PrimingAmount']]
# Visualize the results
missing_by_brewmethod.plot(kind='bar', figsize=(12, 8))
plt.title('Proportion of Missing Data by Brew Method')
plt.ylabel('Proportion Missing')
plt.xlabel('Brew Method')
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.legend(title='Variable')
plt.show(), missing_by_brewmethod

for column in ['PrimaryTemp', 'PitchRate']:
    df[column] = df.groupby('BrewMethod')[column].transform(lambda x: x.fillna(x.median()))

print(df[['PrimaryTemp', 'PitchRate']].isnull().sum())

df.loc[df['BrewMethod'] == 'All Grain', 'MashThickness'] = df.loc[df['BrewMethod'] == 'All Grain', 'MashThickness'].fillna
(df[df['BrewMethod'] == 'All Grain']['MashThickness'].median())

df.loc[df['BrewMethod'] != 'All Grain', 'MashThickness'] = np.nan

print(df['MashThickness'].isnull().sum(), df['MashThickness'].notnull().sum())

df['PrimingMethod'] = df['PrimingMethod'].fillna('None')


