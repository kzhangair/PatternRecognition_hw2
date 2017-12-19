import numpy as np
from scipy import io
from sklearn import svm
from sklearn import preprocessing
from sklearn.decomposition import PCA

def Main():
    fo = io.loadmat("trainingData.mat")
    trainingData = fo['trainingData']
    fo = io.loadmat("testingData.mat")
    testingData = fo['testingData']
    X = trainingData[:, :-1]
    pca = PCA(n_components=15)
    pca.fit(X)
    X_pca = pca.transform(X)
    X_scaled = preprocessing.scale(X_pca)
    Y = trainingData[:, -1]
    testX = testingData[:, :-1]
    testX_pca = pca.transform(testX)
    testX_scaled = preprocessing.scale(testX_pca)
    testY = testingData[:, -1]
    C_space = [0.7]
    Gamma_space = [0.12]

    for i in range(1):
        for j in range(1):
            clf = svm.SVC(C=C_space[i], gamma=Gamma_space[j])
            print "C = %f gamma = %f begin training" % (C_space[i], Gamma_space[j])
            print "shape is : %d %d" % (X_scaled.shape[0], X_scaled.shape[1])
            clf.fit(X_scaled, Y)
            print "begin predicting"
            R = clf.predict(testX_scaled)
            len = R.shape[0]
            count = 0
            for k in range(len):
                if R[k] == testY[k]:
                    count = count + 1
            accuracy = float(count) / len
            print "rbf Kernel with C = %f , gamma = %f, Accuracy is %f" % \
                  (C_space[i], Gamma_space[j], accuracy)
