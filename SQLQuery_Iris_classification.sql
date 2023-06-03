DECLARE @classes NVARCHAR(MAX);
DECLARE @length_mean NVARCHAR(MAX);
DECLARE @length_variance NVARCHAR(MAX);
DECLARE @width_mean NVARCHAR(MAX);
DECLARE @width_variance NVARCHAR(MAX);

SELECT @classes = STRING_AGG(class, ';') WITHIN GROUP (ORDER BY id) 
FROM dbo.data;
SELECT @length_mean = STRING_AGG([length mean], ';') WITHIN GROUP (ORDER BY id) 
FROM dbo.data;
SELECT @length_variance = STRING_AGG([length variance], ';') WITHIN GROUP (ORDER BY id) 
FROM dbo.data;
SELECT @width_mean = STRING_AGG([width mean], ';') WITHIN GROUP (ORDER BY id) 
FROM dbo.data;
SELECT @width_variance = STRING_AGG([width variance], ';') WITHIN GROUP (ORDER BY id) 
FROM dbo.data;

DECLARE @sepal_length NVARCHAR(MAX);
DECLARE @sepal_width NVARCHAR(MAX);
DECLARE @y_input NVARCHAR(MAX);

SELECT @sepal_length = STRING_AGG([sepal length (cm)], ';') WITHIN GROUP (ORDER BY id) 
FROM dbo.Input;
SELECT @sepal_width = STRING_AGG([sepal width (cm)], ';') WITHIN GROUP (ORDER BY id)
FROM dbo.Input;
SELECT @y_input = STRING_AGG(class, ';') WITHIN GROUP (ORDER BY id)
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

cls = classes.split(";")
len_mean = np.fromstring(length_mean, dtype=float, sep=";")
len_var = np.fromstring(length_variance, dtype=float, sep=";")
wid_mean = np.fromstring(width_mean, dtype=float, sep=";")
wid_var = np.fromstring(width_variance, dtype=float, sep=";")
coef_data = list(zip(cls,len_mean,len_var,wid_mean,wid_var))

mean_per_class = np.array(list(zip(len_mean, wid_mean)))
var_per_class = np.array(list(zip(len_var, wid_var)))
print("mean per class:\n",mean_per_class,"\n")
print("variance per class:\n",var_per_class,"\n")

sepal_len = np.fromstring(sepal_length, dtype=float, sep=";")
sepal_wid = np.fromstring(sepal_width, dtype=float, sep=";")
X_data = list(zip(sepal_len, sepal_wid))
y_data = np.fromstring(y_input, dtype=float, sep=";")

X_train, X_test, y_train, y_test = train_test_split(X_data,y_data,test_size=0.2)

log_prob = []
for i in range(len(X_test)):
    log_prob_i = []
    for j in range(len(mean_per_class)):
        log_prob_i.append(np.sum(np.log(np.exp(-0.5 * ((X_test[i] - mean_per_class[j])**2 / var_per_class[j])) / np.sqrt(2 * np.pi * var_per_class[j]))))
    log_prob.append(log_prob_i)


y_pred = np.argmax(log_prob, axis=1)
print("y test: \n", y_test.astype(np.int), "\n")
print("y predicted: \n", y_pred, "\n")


def get_color(y):
    if y > 0:
        return "blue"
    else:
        return "red"

# Map the colors to the y-values using the get_color function
colors = [get_color(val) for val in y_pred]

print("Class 1:", list(y_pred).count(0), " elements\n")
print("Class 2:", list(y_pred).count(1), " elements\n")

from matplotlib.lines import Line2D

legend_elements = [Line2D([0], [0], marker="o", color="w", label="Class 1",
                          markerfacecolor="red", markersize=5),
                   Line2D([0], [0], marker="o", color="w", label="Class 2",
                          markerfacecolor="blue", markersize=5)]

# Create the figure
fig, ax = plt.subplots()

ax.legend(handles=legend_elements)

ax.scatter(np.array(X_test)[:,0],np.array(X_test)[:,1], c=colors)
ax.set_xlabel("Sepal length")
ax.set_ylabel("Sepal width")
plt.savefig("C:/sql_plots/lab4_class.png", format="png")
',
@params = N'@classes NVARCHAR(MAX), @length_mean NVARCHAR(MAX),
@length_variance NVARCHAR(MAX), @width_mean NVARCHAR(MAX), @width_variance NVARCHAR(MAX),
@sepal_length NVARCHAR(MAX), @sepal_width NVARCHAR(MAX), @y_input NVARCHAR(MAX)',
@classes = @classes,
@length_mean = @length_mean,
@length_variance = @length_variance,
@width_mean = @width_mean,
@width_variance = @width_variance,
@sepal_length = @sepal_length,
@sepal_width = @sepal_width,
@y_input = @y_input
GO
_length = @sepal_length,
@sepal_width = @sepal_width,
@y_input = @y_input
GO
