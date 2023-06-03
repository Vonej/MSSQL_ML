USE Iris_db;
GO
CREATE PROCEDURE insert_iris_data AS
BEGIN
INSERT INTO Iris_data([sepal length (cm)], [sepal width (cm)],[petal length (cm)],[petal width (cm)],[class label],species)
EXEC sp_execute_external_script  @language =N'Python',
@script=N'
import pandas as pd
from sklearn.datasets import load_iris
import numpy as np

iris = load_iris()

df = pd.DataFrame(data=np.c_[iris["data"], iris["target"]],
                  columns= iris["feature_names"] + ["target"]).astype({"target": int})

df = df.assign(species=lambda x: x["target"].map(dict(enumerate(iris["target_names"]))))

df["species"] = df["species"].astype(str)

OutputDataSet = df
'
END;