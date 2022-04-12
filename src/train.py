from .utils import *
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import RegressionEvaluator


def train_model(data, regressor, labelCol, params={}):
    """
    Returns the trained pipeline and Spark DataFrame with relevant inputs and predictions

    Parameters
    ----------
    data : Spark DataFrame
        The input DataFrame
    regressor : obj
        The regression model object from MLlib
    labelCol : str
        The name of the target column in the DataFrame.
    params : dict, optional
        Parameters for the PySpark regression model as key-value pairs
    """
    
    pollutant_mean = labelCol.split()[0] + ' Mean'
    inputCols = ['Year', 'Month', 'Day', 'County_index', 'City_index', pollutant_mean]
    
    train, test = create_splits(data, labelCol)
    
    indexers = [StringIndexer(inputCol=column, outputCol=column+"_index").fit(train) for column in ['County', 'City']]
    assembler = VectorAssembler(inputCols=inputCols,
                          outputCol='features')
    m = regressor(labelCol=labelCol, **params)
    
    stages = indexers + [assembler, m]
    pipeline = Pipeline(stages=stages)
    
    model = pipeline.fit(train)  
    preds_ts = model.transform(test)
    
    evaluator = RegressionEvaluator(labelCol=labelCol, predictionCol="prediction", metricName="rmse")
    test_rmse = evaluator.evaluate(preds_ts)
   
    print(f"Root Mean Squared Error (RMSE) on test data = {test_rmse}")
    print()
    
    return model, preds_ts[inputCols + ['Date', labelCol, 'features', 'prediction', 'City']]