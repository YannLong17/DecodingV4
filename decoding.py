import data
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cross_validation import StratifiedKFold, StratifiedShuffleSplit
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import make_scorer
from sklearn.decomposition import PCA

from sklearn.naive_bayes import PoissonNB

def getCoordinate(a):
    x = a % 10
    y = a // 10
    return x,y

def distance (a,b):
    x1,y1 = getCoordinate(a)
    x2,y2 = getCoordinate(b)
    d = (abs(x2 - x1) + abs(y2 - y1)) / (((x2 - x1) ** 2 + (y2 - y1) ** 2) ** (1 / 2))
    return d

def laxScore (y, y_pred, d):
    #To Do: verify that len(y) = len(y_pred)

    acc=[]
    for i in range(0,len(y)):
        if distance(y[i],y_pred[i]) < d: acc.append(1)
        else: acc.append(0)

    return np.mean(np.asarray(acc))

def dimReduction():
    file = 'n228_bcdefgh.mat'
    dat = data.load(file)

    # Dimentionnality Reduction
    # Select good cell with heuristic
    channel = data.goodCell(dat)
    X, Y = data.build(dat, channel, 'fr1', 17)

    # let PCA do the work
    Xa, Ya = data.build(dat, range(0, 96), 'fr1', 17)
    pca = PCA(n_components=38)
    Xb = pca.fit_transform(Xa)

    # Cross Validation
    skf = StratifiedKFold(Y, n_folds=5, shuffle=True, random_state=42)

    d = range(0, 25)
    train_acc = []
    test_acc = []
    traina_acc = []
    testa_acc = []
    trainb_acc = []
    testb_acc = []
    chance = []

    for i in d:
        train_temp = []
        test_temp = []
        traina_temp = []
        testa_temp = []
        trainb_temp = []
        testb_temp = []
        for train_index, test_index in skf:
            Y_train, Y_test = Y[train_index], Y[test_index]

            # heuristic
            X_train, X_test = X[train_index], X[test_index]
            learner = GaussianNB().fit(X_train, Y_train)
            train_temp.append(laxScore(Y_train, learner.predict(X_train), i))
            test_temp.append(laxScore(Y_test, learner.predict(X_test), i))

            # No reduction
            Xa_train, Xa_test = Xa[train_index], Xa[test_index]
            learner = GaussianNB().fit(Xa_train, Y_train)
            traina_temp.append(laxScore(Y_train, learner.predict(Xa_train), i))
            testa_temp.append(laxScore(Y_test, learner.predict(Xa_test), i))

            # PCA
            Xb_train, Xb_test = Xb[train_index], Xb[test_index]
            learner = GaussianNB().fit(Xb_train, Y_train)
            trainb_temp.append(laxScore(Y_train, learner.predict(Xb_train), i))
            testb_temp.append(laxScore(Y_test, learner.predict(Xb_test), i))

        # chance
        temp = []
        for k in range(0, 100):
            tempy = []
            for j in range(0, 100):
                if distance(j, k) < i:
                    tempy.append(1)
                else:
                    tempy.append(0)
            temp.append(np.mean(np.asarray(tempy)))

        chance.append(np.mean(np.asarray(temp)))
        train_acc.append(np.mean(np.asarray(train_temp)))
        test_acc.append(np.mean(np.asarray(test_temp)))
        traina_acc.append(np.mean(np.asarray(traina_temp)))
        testa_acc.append(np.mean(np.asarray(testa_temp)))
        trainb_acc.append(np.mean(np.asarray(trainb_temp)))
        testb_acc.append(np.mean(np.asarray(testb_temp)))


        #    plt.plot(d,train_acc,linewidth=2.0)
    plt.plot(d, test_acc, linewidth=2.0)
    #    plt.plot(d,traina_acc,linewidth=2.0)
    plt.plot(d, testb_acc, linewidth=2.0)
    plt.plot(d, testa_acc, linewidth=2.0)
    plt.plot(d, chance)
    plt.ylabel('Accuracy')
    plt.xlabel('Distance')
    plt.legend(['Heuristic', 'PCA', 'No reduction', 'Chance'], loc='upper right')
    plt.show()
    #    plt.savefig('n228/Fr1AccDimRed.png')

def localization():
    file = 'n191_bcde.mat'
    dat = data.load(file)

    # Dimentionnality Reduction
    X, y = data.build(dat, range(0, 96), 'fr1', 17)
    pca = PCA(n_components=38)
    X = pca.fit_transform(X)

    # Cross Validation
    split = StratifiedShuffleSplit(y, n_iter=1, test_size=0.1, random_state=42)

    for train_index, test_index in split:
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

    learner = GaussianNB().fit(X_train, y_train)
    y_pred = learner.predict(X_test)

    arrowPlot(y_test,y_pred)

def arrowPlot(y,y_pred):
    pred=np.zeros((2,100))
    real=np.zeros((2,100))

    for i in range(0,100):
        temp=[]
        for ele in np.nditer(y_pred[y==i]):
            temp.append(np.array(getCoordinate(ele)).reshape(2,1))
        pred[:,i] = np.mean(np.asarray(temp), axis=0)[:,0]
        real[:,i] = np.array(getCoordinate(i)).reshape(2,1)[:,0]

#        plt.plot([real[0,i],pred[0,i]],[real[1,i],pred[1,i]])
    origin = np.zeros(len(real[0,:]))

    plt.subplot(2, 1, 1)
    plt.quiver(real[0,:],real[1,:],(pred[0,:]-real[0,:]),(pred[1,:]-real[1,:]))
    plt.subplot(2, 1, 2)
    plt.quiver(origin, origin, (pred[0, :] - real[0, :]), (pred[1, :] - real[1, :]))

#    plt.show()
    plt.savefig('m191Fr1Localization.png')

if __name__ == '__main__':
    localization()