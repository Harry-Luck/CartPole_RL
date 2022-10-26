import pandas, scipy, numpy
from sklearn.preprocessing import MinMaxScaler
df=pandas.read_csv( 'http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv ',sep=';')
array=df.values
#Separating data into input and output components<font></font>
x=array[:,0:8]
y=array[:,8]
scaler=MinMaxScaler(feature_range=(0,1))
rescaledX=scaler.fit_transform(x)
numpy.set_printoptions(precision=3) #Setting precision for the output<font></font>
rescaledX[0:5,:]