import matplotlib.pyplot as plt

def plot_splits(pollutant, train, test):
    """
    Plot the time-series train and test splits

    Parameters
    ----------
    pollutant : str
        Name of the pollutant column
    train : Spark DataFrame
        The train dataframe
    test : str
        The test DataFrame
    """
    
    pd_train = train.groupby('Date').agg({pollutant: 'mean'}).orderBy('Date').toPandas().set_index('Date')
    pd_test = test.groupby('Date').agg({pollutant: 'mean'}).orderBy('Date').toPandas().set_index('Date')
    
    plt.figure(figsize=(14,5))
    plt.plot(pd_train.index, pd_train[f'avg({pollutant})'], label='Train')
    plt.plot(pd_test.index, pd_test[f'avg({pollutant})'], color='C1', label='Test')
    plt.legend()

    plt.title('Train/Test Sets');
    

def create_splits(data, target_col, plot=False):
    """
    Splits and returns the dataset into train and test

    Parameters
    ----------
    data : Spark DataFrame
        The input DataFrame
    target_col : str
        The name of the target column
    plot : boolean, optional
        If True, it calls the plot_split function to plot the dataset splits
    """
    train = data[(data['Date'] > '2015-01-01') & (data['Date'] <= '2021-01-01')]
    test = data[data['Date'] > '2021-01-01']
    
    if plot:
        plot_splits(target_col, train, test)
        return None
    
    return (train, test)


def plot_city_preds(pollutant, data, city):
    """
    Plots the predictions obtained by the model based on a city

    Parameters
    ----------
    pollutant : str
        Name of the pollutant column
    data : Spark DataFrame
        The input DataFrame
    city : str
        Name of the City to plot
    """
        
    data = data[data['City'] == city]
    pred = data.groupby('Date').agg({'prediction': 'mean'}).orderBy('Date').toPandas().set_index('Date')
    actual = data.groupby('Date').agg({pollutant: 'mean'}).orderBy('Date').toPandas().set_index('Date')
    
    plt.figure(figsize=(14,5))
    plt.plot(pred.index, pred[f'avg(prediction)'], label='Predictions')
    plt.plot(actual.index, actual[f'avg({pollutant})'], color='C1', label='Actual', linestyle='--')
    plt.xlabel('Year', fontsize=18)
    plt.ylabel('AQI', fontsize=18)
    plt.legend()

    plt.title(f'Predictions Vs Actual - {city} ({pollutant})', fontsize=20);
