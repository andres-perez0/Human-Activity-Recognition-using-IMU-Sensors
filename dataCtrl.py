# for data processing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# for number - crunching
import numpy as np

# for dataset management
import pandas as pd

class dataCtrl():
    def __init__(self):
        pass
    # The function does use 'self'
    @staticmethod
    def initialize_data() -> pd.DataFrame:
        # read data from the combine csv
        data = pd.read_csv('combine_mpu9250.csv',header=0,on_bad_lines='skip',sep=',')

        # imputing NaN Values
        sensor_cols=['accX', 'accY', 'accZ', 'gyroX', 'gyroY', 'gyroZ']
        for col in sensor_cols:
            # fills the columns with the average of the column
            data[col] = data[col].fillna(data[col].mean())
        
        # print(data.isnull().sum()) # verifies the imputing values worked
        return data

    @staticmethod
    def sliding_window(data: pd.DataFrame, window_size: int, overlap_percentage: float):
        # Data columns to be segmented
        sensor_cols=['accX', 'accY', 'accZ', 'gyroX', 'gyroY', 'gyroZ']

        # Step size based on window size and overlap
        step_size = int(window_size * (1-overlap_percentage))

        if step_size <= 0:
            raise ValueError("calculated step size is zero or negative")
        
        windows_with_labels=[]

        # Iterate through the df in steps
        for i in range(0, len(data) - window_size + 1, step_size):
            window       = data.iloc[i:i+window_size]
            window_data  = window[sensor_cols].values
            window_label = window['activity_label'].mode()[0]

            # Saves the data and activity as a dictionary and appends it to the list
            windows_with_labels.append({'data':window_data, 'activity_label': window_label})

        return windows_with_labels

    def preprocess_and_spilt(self, data: pd.DataFrame, window_size: int, overlap_percentage: float, test_size:float=0.2):
        raw_segments = self.sliding_window(data,window_size,overlap_percentage)

        # Separate raw data and labels for further processing
        X_raw = [s['data'] for s in raw_segments]
        y_labels = [s['activity_label'] for s in raw_segments]

        X_standardized=[]
        # Initializes the Standard Scaler, it will be fit for each window.
        # The scaler should only see the data it is transforming to avoi data leaks from the future
        scaler=StandardScaler()

        for segment in X_raw:
            # segment is a 2D array
            standardized_segment = scaler.fit_transform(segment)
            X_standardized.append(standardized_segment)
        # converts the list to an np array
        X_standardized = np.array(X_standardized)

        # Converts labels into integers using their key
        label_to_int = {'sitting': 0, 'walking': 1}
        y_encoded = np.array([label_to_int[i] for i in y_labels])

        train_data, test_data, train_labels, test_labels = train_test_split(
            X_standardized, y_encoded, test_size=test_size, stratify=y_encoded, random_state=42
        )
        return train_data, test_data, train_labels, test_labels

    