import tensorflow.keras as kr
import numpy as np

import pandas as pd

veri=pd.read_csv("C:/Users/user/Downloads/Neural-Network-Example-master/iris.data")

veri=veri.replace("Iris-setosa",0)
veri=veri.replace("Iris-versicolor",1)
veri=veri.replace("Iris-virginica",2)

giris=veri.iloc[:,0:4]
cikis=veri.iloc[:,4:]

model=kr.Sequential()
model.add(kr.layers.Dense(64,input_dim=4))
model.add(kr.layers.Activation("relu"))
model.add(kr.layers.Dense(3))
model.add(kr.layers.Activation("softmax"))

model.compile(optimizer=kr.optimizers.Adadelta(),loss="sparse_categorical_crossentropy",metrics=["accuracy"])
model.fit(giris,cikis,epochs=10,validation_split=0.13)

deneme1=np.array([4.9,3,1.4,0.2]).reshape(1,4)#0
deneme2=np.array([6.1,2.3,4.1,1.2]).reshape(1,4)#1

print(model.predict_classes(deneme1))
print(model.predict_classes(deneme2))
a=np.array([6.7,3.3,5.7,2.5]).reshape(1,-1)
print(model.predict_classes(a))
