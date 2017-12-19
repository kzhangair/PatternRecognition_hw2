import numpy as np
from scipy import io
from sklearn import svm
from sklearn import preprocessing
import threading
from sklearn.model_selection import cross_val_score

fo = io.loadmat("trainingData.mat")
trainingData = fo['trainingData']
fo = io.loadmat("testingData.mat")
testingData = fo['testingData']
X = trainingData[:, :-1]
X_scaled = preprocessing.scale(X)
Y = trainingData[:, -1]
testX = testingData[:, :-1]
testX_scaled = preprocessing.scale(testX)
testY = testingData[:, -1]
C_space = np.logspace(-5, 15, num=5, base=2, dtype=float)
Gamma_space = np.logspace(-15, 3, num=5, base=2, dtype=float)

class CrossVal(threading.Thread):
    def __init__(self, C_index, Gamma_index):
        threading.Thread.__init__(self)
        self.C_index = C_index
        self.Gamma_index = Gamma_index
    def run(self):
        clf = svm.SVC(C=C_space[self.C_index], kernel='rbf', gamma=Gamma_space[self.Gamma_index] ,class_weight='balanced')
        print "C = %f begin training" % C_space[self.C_index]
        clf.fit(X_scaled, Y)
        print "C = %f begin predicting" % C_space[self.C_index]
        R = clf.predict(testX_scaled)
        len = R.shape[0]
        count = 0
        for i in range(len):
            if R[i] == testY[i]:
                count = count + 1
        accuracy = float(count) / len
        print "rbf Kernel with C = %f , gamma = %f, Accuracy is %f" % \
              (C_space[self.C_index], Gamma_space[self.Gamma_index], accuracy)

for i in range(5):
    for j in range(5):
        t = CrossVal(i, j)
        t.start()


