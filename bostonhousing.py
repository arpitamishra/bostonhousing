from sklearn import datasets
import numpy as np
import pandas as pd
import statsmodels.api as sm
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
model.summary()
predictions.to_csv("C:\\Users\\RAJIV MISHRA\\Desktop\\Arpita\\sampleproblemdataset\\housing data\\output.csv")
