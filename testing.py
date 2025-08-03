# Counting the number of rows in a csv file
import pandas as pd

df=pd.read_csv('IMU_Data_2\mpu9250_data15.csv')
csvLength = len(df)

print(f"the df is {csvLength} rows.\nthe df shape is {df.shape}")

# Checking for missing values in rows
missingValuesPercolumn=df.isnull().sum()

print("Missing values per column:")
print(missingValuesPercolumn)

incompleteRowMask=df.isnull().any(axis=1)

incompleteRowCount=incompleteRowMask.sum()

print('\n Incomplete Rows')
print(df[incompleteRowMask])

df_filtered = df [ incompleteRowMask != True]

print(df_filtered)