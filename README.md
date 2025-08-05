## Take Always
- Static Classification Models vs Time-based classification models
    - SCMs are designed to classify data points indepenedently of any temporal context. Aka the order which data is processed does not matter, think about the quality of a meal, it has calories, filling-ness, saturation, taste, and more. If you were making a model to identity the quality of a meal, whether or not you test a sandwich or soup does not matter. 

    - Data points are independent and identicially distributed
    - Common Models (according to gemini): Logisitc Regression, Support Vector Machines (SVMs), Decision Trees and Random Forest, Standard Feed-forward neural Networks (MLPs)

- Time-Based Classification Models 
    - TBCs are design to handle data where full context is cruical for making a correct classifications. So think about a sequence or time series of data points. Think Speech Recognitions, Video Analysis, Natural Language Processing, and Financial Time Series. 
    - In this model, this can be found in the sliding window chunking of data, where we chunk data based on entries, to present the walking motion. Although admittedly. If I kept a constantly Hz output from the arduino and made a form of error checking, the data would produce slightly more accurate predicition. 
    - Common Models include Recurrent Neural Networks, Long Short-Term Memory, Temporal Convolutional Networks, and Transformers

- Windowing or Sliding Window Segmentation
    - This is the general name for the technique of dividing a long time series into smaller, overlapping or non-overlapping segments (the "windows").
    - The sliding window function creates a list of individual segments. Thus the model architecture has to change, to accept these segments and further in batches. 
    - As you can see below the general jist of the algorithm, once again explained by gemini, is to step based on the window size and overlap_percentage (to give it a 'memory' between steps) throughout the dataframe and take row slives to further extrapolate from;
    ```python
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

    # print(len(window_label))
    return windows_with_labels
    ```
- Time-Series Analysis

- Standard Scalers 
    - A pre-processing step in ml, particularly for algorithms that are sensitive to the scale and distribution of data. Its purpose is to standardize features by transforming them to have a mean of 0 and std. dev. of 1

    - In this case, you will feel a greater change in g (meters per seconds) while walking than let's say sitting or standing in place. This can  caused a bias and less accurate model since the model might incorrectly interpret the differences min.s and max.s of both activity as drastically differing weights. 

    - Normalization of any sort is generally good for model performance because they are able to converge faster when the features are scaled. Take Logisitic Regression, both L1 and L2, are center around zero with similar variance. 

- Simple verus Deep Learning Models
    - I went crazy with why I was getting high 50 percentage in my models. Until, I took a step to look back. The data I have right now is pretty black and white, so any simple linear regression can do fairly well. Or, I have yet to learn more about decision trees, in particular Random Forest, so I have a lot more thinkering until I gain an intuition with DP, but it'll be fun to do so.

