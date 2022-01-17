from numpy import NaN
import pandas as pd
#import openpyxl
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as si
"""
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, LeakyReLU
from keras import backend
"""

from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, LeakyReLU
from keras import backend

#df = pd.read_csv(r"C:\Users\q8606\Desktop\optiodata.csv", sep=";")

daatta1 = pd.read_csv(r"C:\Users\q8606\Desktop\GRADUTUTKIMUKSET\data1.csv", sep=";")

daatta2 = pd.read_csv(r"C:\Users\q8606\Desktop\GRADUTUTKIMUKSET\data2.csv", sep=";")
daatta3 = pd.read_csv(r"C:\Users\q8606\Desktop\GRADUTUTKIMUKSET\data3.csv", sep=";")
daatta4 = pd.read_csv(r"C:\Users\q8606\Desktop\GRADUTUTKIMUKSET\data4.csv", sep=";")
daatta5 = pd.read_csv(r"C:\Users\q8606\Desktop\GRADUTUTKIMUKSET\data5.csv", sep=";")
daatta6 = pd.read_csv(r"C:\Users\q8606\Desktop\GRADUTUTKIMUKSET\data6.csv", sep=";")
daatta7 = pd.read_csv(r"C:\Users\q8606\Desktop\GRADUTUTKIMUKSET\data7.csv", sep=";")





#wide to long based on date columns
#meltedframe = pd.melt(df, id_vars="Name")

daatta1 = pd.melt(daatta1,id_vars="Name")
daatta2 = pd.melt(daatta2,id_vars="Name")
daatta3 = pd.melt(daatta3,id_vars="Name")
daatta4 = pd.melt(daatta4,id_vars="Name")
daatta5 = pd.melt(daatta5,id_vars="Name")
daatta6 = pd.melt(daatta6,id_vars="Name")
daatta7 = pd.melt(daatta7,id_vars="Name")


combined_data = pd.concat([daatta1,daatta2,daatta3,daatta4,daatta5,daatta6,daatta7])

print(combined_data)
print("yhdistyy")




combined_data = combined_data[~combined_data.variable.str.contains('#ERROR')]

#splitting variable to A and B columns
combined_data[['A', 'B']] = combined_data['variable'].str.split(' - ', 1, expand=True)
combined_data = combined_data[~combined_data.A.str.contains('22')]
meltedframe = combined_data
print(combined_data)
print("jatkuu")


meltedframe.loc[pd.isnull(meltedframe['B']) == True, 'B'] = "CALL_PRICE"
meltedframe.drop('variable', inplace=True, axis=1)
meltedframe.loc[meltedframe['B'].str.contains("OPT STRIKE PRICE"), 'B'] = "STRIKE"
meltedframe.loc[meltedframe['B'].str.contains("OPT.U/LYING PRICE"), 'B'] = "UNDERLYING"
meltedframe.loc[meltedframe['B'].str.contains("IMPLIED VOL."), 'B'] = "IV"

print(meltedframe)
print("testi")
meltedframe = meltedframe.dropna()
print(meltedframe)
print("dropin jälkeen")
print(meltedframe["B"].unique())
print("uniikit")
meltedframe = meltedframe.drop_duplicates()
df1 = meltedframe.pivot(index=["Name","A"], columns="B",values="value").reset_index()
df1 = df1.dropna()
df1 = df1.drop_duplicates()
print(df1)
print(df1["A"].nunique())


#print(meltedframe)
#print("nimeäminen")

#pivoting by unique multi-index date and option name
#df1 = meltedframe.pivot(index=["Name","A"], columns="B",values="value").reset_index()




#statistics






#changing datatypes & cleaning

df1["UNDERLYING"]=df1["UNDERLYING"].str.replace(',','.')
df1["CALL_PRICE"] = df1["CALL_PRICE"].str.replace(',','.')
df1["STRIKE"] = df1["STRIKE"].str.replace(',','.')
df1["IV"]=df1["IV"].str.replace(',','.')



print(df1)
df1["UNDERLYING"] = df1.UNDERLYING.astype("float")
df1["STRIKE"] = df1.STRIKE.astype("float")
df1["CALL_PRICE"] = df1.CALL_PRICE.astype("float")
df1["IV"]=df1.IV.astype("float")
df1['Name']= pd.to_datetime(df1['Name'], format = '%d.%m.%Y')


print(df1.info())

#moneyness
df1["Moneyness"] = df1["STRIKE"] / df1["UNDERLYING"]
df1 = df1.sort_values(["A","Name"])

#time-to-expiration
df2 = df1.groupby('A')['Name'].last()
df1 = df1.merge(df2, how = 'inner', on = ['A'])
df1["TTM"] = df1["Name_y"] - df1["Name_x"]
df1["TTM"] = df1.TTM.dt.days.astype("float")

#removing observations based on hutchinson

#df1 = combined_data[~combined_data.variable.str.contains('#ERROR')]
#ylisataset = df1[(df1['TTM'] > 120)]  
#df1 = df1[~(df1['TTM'] > 120)] 
df1["TTM"] = df1["TTM"] / 365

"""
#drop na values




"""
#df1["UNDERLYING"] = df1["UNDERLYING"]/df1["STRIKE"]
#df1["CALL_PRICE"] = df1["CALL_PRICE"]/df1["STRIKE"]
print(df1)
X=df1[['IV', 'UNDERLYING', 'STRIKE', 'TTM']]
#X = X.astype("int")
y=df1['CALL_PRICE']
#y = y.astype("int")

"""


def blackscholes(S, K, T, r, sigma):
    
    #S: spot price
    #K: strike price
    #T: time to maturity
    #r: interest rate
    #sigma: volatility of underlying asset
    
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = (np.log(S / K) + (r - 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    
    call = (S * si.norm.cdf(d1, 0.0, 1.0) - K * np.exp(-r * T) * si.norm.cdf(d2, 0.0, 1.0))
    
    return call



#df1["bsprice"] = blackscholes(df1["UNDERLYING"],df1["STRIKE"],df1["TTM"],0.05,df1["IV"])

#print(df1)





X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)



nodes = 30
model = Sequential()

model.add(Dense(120, input_dim=4, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
          
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


arvot = model.fit(X_train, y_train, batch_size=10, epochs=10)





model = Sequential()
model.add(Dense(1024, input_dim=X.shape[-1], activation='relu'))
model.add(Dropout(.25))
model.add(Dense(2048, activation='relu'))
model.add(Dropout(.25))
model.add(Dense(1, activation='linear'))
model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, epochs=100, validation_split=.1, verbose=1, batch_size=256)

def CheckAccuracy(y,y_hat):
    stats = dict()
    
    stats['diff'] = y - y_hat
    
    stats['mse'] = np.mean(stats['diff']**2)
    print("Mean Squared Error:      ", stats['mse'])
    
    stats['rmse'] = np.sqrt(stats['mse'])
    print("Root Mean Squared Error: ", stats['rmse'])
    
    stats['mae'] = np.mean(abs(stats['diff']))
    print("Mean Absolute Error:     ", stats['mae'])
    
    stats['mpe'] = np.sqrt(stats['mse'])/np.mean(y)
    print("Mean Percent Error:      ", stats['mpe'])


y_train_hat = model.predict(X_train)
#reduce dim (240000,1) -> (240000,) to match y_train's dim
y_train_hat = np.squeeze(y_train_hat)
CheckAccuracy(y_train, y_train_hat)
model.evaluate(X_test, y_test)
"""

def CheckAccuracy(y,y_hat):
    stats = dict()
    
    stats['diff'] = y - y_hat
    
    stats['mse'] = np.mean(stats['diff']**2)
    print("Mean Squared Error:      ", stats['mse'])
    
    stats['rmse'] = np.sqrt(stats['mse'])
    print("Root Mean Squared Error: ", stats['rmse'])
    
    stats['mae'] = np.mean(abs(stats['diff']))
    print("Mean Absolute Error:     ", stats['mae'])
    
    stats['mpe'] = np.sqrt(stats['mse'])/np.mean(y)
    print("Mean Percent Error:      ", stats['mpe'])


#CheckAccuracy(df1["bsprice"],df1["CALL_PRICE"])


def blackscholes(S, K, T, r, sigma):
    
    #S: spot price
    #K: strike price
    #T: time to maturity
    #r: interest rate
    #sigma: volatility of underlying asset
    
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = (np.log(S / K) + (r - 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    
    call = (S * si.norm.cdf(d1, 0.0, 1.0) - K * np.exp(-r * T) * si.norm.cdf(d2, 0.0, 1.0))
    
    return call



df1["bsprice"] = blackscholes(df1["UNDERLYING"],df1["STRIKE"],df1["TTM"],0.05,df1["IV"])
df1["bsprice"]=df1["bsprice"].round(4)
print(df1)
print(df1.info())
print(df1["TTM"].mean(), "mean value")
print(df1["TTM"].max(), "max value")

