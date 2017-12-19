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
C_space = np.logspace(-5, 15, num=10, base=2, dtype=float)

class CrossVal(threading.Thread):
    def __init__(self, C_index):
        threading.Thread.__init__(self)
        self.C_index = C_index
    def run(self):
        clf = svm.LinearSVC(C=C_space[self.C_index], class_weight='balanced')
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
        print "Linear Kernel with C = %f , Accuracy is %f" % (C_space[self.C_index], accuracy)


t0 = CrossVal(0)
t0.start()
t1 = CrossVal(1)
t1.start()
t2 = CrossVal(2)
t2.start()
t3 = CrossVal(3)
t3.start()
t4 = CrossVal(4)
t4.start()
t5 = CrossVal(5)
t5.start()
t6 = CrossVal(6)
t6.start()
t7 = CrossVal(7)
t7.start()
t8 = CrossVal(8)
t8.start()
t9 = CrossVal(9)
t9.start()


