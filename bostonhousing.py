from sklearn import datasets
import numpy as np
import pandas as pd
import statsmodels.api as sm
def data_laoder_and_predict_function():
  data = datasets.load_boston()
  # define the data/predictors as the pre-set feature names
  df = pd.DataFrame(data.data, columns=data.feature_names)
  # Put the target (housing value -- MEDV) in another DataFrame
  target = pd.DataFrame(data.target, columns=["MEDV"])
  X = df["RM"]
  y = target["MEDV"]
  # Note the difference in argument order
  model = sm.OLS(y, X).fit()
  predictions = model.predict(X) # make the predictions by the model
  # Print out the statistics
  summary = model.summary()
  print(summary)
  predictions.to_csv(".\sampleproblemdataset\\housing data\\output.csv")
  
  if __name__=="__main__":
    data_loader_and_predict_function()
