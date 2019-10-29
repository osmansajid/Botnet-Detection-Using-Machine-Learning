import LoadData
import DataPreparation
import pickle
import warnings
warnings.filterwarnings("ignore")
LoadData.loaddata('flowdata.binetflow')
file = open('flowdata.pickle', 'rb')
data  = pickle.load(file)
Xdata = data[0]
Ydata =  data[1]
XdataT = data[2]
YdataT = data[3]
from sklearn.linear_model import *
from sklearn.tree import *
from sklearn.naive_bayes import *
from sklearn.neighbors import *
DataPreparation.Prepare(Xdata,Ydata,XdataT,YdataT)
clf = DecisionTreeClassifier()
clf.fit(Xdata,Ydata)
Prediction = clf.predict(XdataT)
Score = clf.score(XdataT,YdataT)
print ("The Score of the Decision Tree Classifier is", Score * 100)
clf = LogisticRegression(C=10000)
clf.fit(Xdata,Ydata)
Prediction = clf.predict(XdataT)
Score = clf.score(XdataT,YdataT)
print ("The Score of the Logistic Regression Classifier is", Score * 100)
clf = GaussianNB()
clf.fit(Xdata,Ydata)
Prediction = clf.predict(XdataT)
Score = clf.score(XdataT,YdataT)
print("The Score of the Gaussian Naive Bayes classifier is", Score * 100)
clf = KNeighborsClassifier()
clf.fit(Xdata,Ydata)
Prediction = clf.predict(XdataT)
Score = clf.score(XdataT,YdataT)
print("The Score of the K-Nearest Neighbours classifier is", Score * 100)
from keras.models import *
from keras.layers import Dense, Activation
from keras.optimizers import *
model = Sequential()
model.add(Dense(10, input_dim=9, activation="sigmoid"))
model.add(Dense(10, activation='sigmoid'))
model.add(Dense(1))
sgd = SGD(lr=0.01, decay=0.000001, momentum=0.9, nesterov=True) 
model.compile(optimizer=sgd, loss='mse')
model.fit(Xdata, Ydata, nb_epoch=200, batch_size=100)
Score = model.evaluate(XdataT, YdataT, verbose=0)
print("The Score of the Neural Network is", Score * 100 )
