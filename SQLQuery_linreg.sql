DECLARE @x_input NVARCHAR(MAX);
DECLARE @y_input NVARCHAR(MAX);

SELECT @x_input = STRING_AGG(x, ';') WITHIN GROUP (ORDER BY id) 
FROM dbo.Input;
SELECT @y_input = STRING_AGG(y, ';') WITHIN GROUP (ORDER BY id)
FROM dbo.Input;

EXEC sp_execute_external_script  
@language =N'Python',
@script=N'
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

x_data = np.fromstring(x_input, dtype=float, sep=";")
y_data = np.fromstring(y_input, dtype=float, sep=";")

X_train, X_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2)

regr = linear_model.LinearRegression()

regr.fit(X_train.reshape(-1,1), y_train.reshape(-1,1))

y_pred = regr.predict(X_test.reshape(-1,1))

nrmse = np.sqrt(mean_squared_error(y_test, y_pred)/np.var(y_test))

# MSE
print("Normalized root mean squared error: %.2f" % nrmse)
# Determination coef
print("Coefficient of determination: %.2f" % r2_score(y_test, y_pred))


',
@params = N'@x_input NVARCHAR(MAX), @y_input NVARCHAR(MAX)',
@x_input = @x_input,
@y_input = @y_input
GO