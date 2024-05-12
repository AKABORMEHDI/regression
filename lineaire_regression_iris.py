import pandas as pd
from sklearn.linear_model import LinearRegression
#data
df = pd.read_csv('C:/Users/AM/Documents/IA/data iris/Iris.csv')
df.head()

x = df[["SepalLengthCm"]]
y = df["SepalWidthCm"]
#instancier le model
model_linReg = LinearRegression()
#entrainement de model
model_linReg.fit(x,y)
#precision du model
precision = model_linReg.score(x,y)
print(precision*100)
#prediction
longuer = 2.5
prediction = model_linReg.predict([[longuer]])
print(prediction)