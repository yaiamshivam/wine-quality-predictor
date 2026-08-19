import pandas as pd
df = pd.read_csv("WineQT - WineQT.csv.csv")
print(df.head())
print('\n')
print(df.shape)
print('\n')
print(df.columns)
print('\n')
print(df.info())
print('\n')
print(df.describe())
print('\n')
print(df.isnull().sum())
#from sklearn.ensemble import RandomForestClassifier

#model = RandomForestClassifier()
# model.fit(X_train, y_train)
import matplotlib.pyplot as plt

df.hist()
plt.show()
# Check duplicates
print(df.duplicated().sum())

# Remove duplicates if any
data = df.drop_duplicates()

# Check class distribution
print(df['quality'].value_counts())
