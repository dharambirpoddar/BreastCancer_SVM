from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC


bcancer = datasets.load_breast_cancer()

X = bcancer.data
Y = bcancer.target

scaler = StandardScaler();
X = scaler.fit_transform(X) 


Xtrain, Xtest, Ytrain, Ytest \
= train_test_split(X, Y, test_size = 0.25) 

# Linear SVM
svmc=SVC(kernel='linear')
svmc.fit(Xtrain,Ytrain)
Ypred=svmc.predict(Xtest)

svmcscore = accuracy_score(Ypred,Ytest)
print('Accuracy score of Linear SVM Classifier is',100*svmcscore,'%\n')

# Kernel SVM RBF
ksvmc=SVC(kernel='rbf')
ksvmc.fit(Xtrain, Ytrain)
Ypred=ksvmc.predict(Xtest)
svmcscore=accuracy_score(Ypred,Ytest)
print('Accuracy score of kernal SVM classifier with RBF is',100*svmcscore,'%\n')




