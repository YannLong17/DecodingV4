import data
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cross_validation import StratifiedKFold, StratifiedShuffleSplit, permutation_test_score
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import make_scorer
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, chi2, f_classif
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV

from NaiveBayes import PoissonNB

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

def error_distance (y, y_pred):
    dist=[]
    for i in range(0,len(y)):
        dist.append(distance(y[i], y_pred[i]))

    return (np.mean(np.asarray(dist)))

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

def model_selection():
    file = 'data/n228_bcdefgh.mat'
    dat = data.load(file)

    X, y = data.build(dat, range(0, 96), 'fr1', 17)

    filter = SelectKBest(chi2, k=5)
    clf = PoissonNB()

    poisson = Pipeline([('filter',filter),('pois',clf)])

    #poisson.set_params(filter__k=10).fit(X,y)

    param_grid = [{'filter__score_func': [chi2], 'filter__k': range(1, 96)},
                  {'filter__score_func': [f_classif], 'filter__k': range(1,96)}]
    grid = GridSearchCV(poisson, param_grid, n_jobs=-1, scoring=make_scorer(error_distance, greater_is_better=False)).fit(X,y)

    print "Best Params"
    print grid.best_params_
    print("Grid scores on development set:")
    print()
    for params, mean_score, scores in grid.grid_scores_:
        print("%0.3f (+/-%0.03f) for %r"
              % (mean_score, scores.std() * 2, params))
    print()

def visualize_model_selection():
    file = 'data/n228_bcdefgh.mat'
    dat = data.load(file)

    X, y = data.build(dat, range(0, 96), 'fr1', 17)

    # PCA Dimentionnality Reduction
    pca = PCA(n_components=38)
    Xb = pca.fit_transform(X)

    # Select good cell with heuristic
    channel = data.goodCell(dat)
    Xc, yc = data.build(dat, channel, 'fr1', 17)

    # Univariate Feature Selection
    select = SelectKBest(chi2,k=15).fit(X,y)
    Xd = select.transform(X)

    select = SelectKBest(f_classif,k=26).fit(X,y)
    Xe = select.transform(X)

    # Cross Validation
    skf = StratifiedKFold(y, n_folds=5, shuffle=True, random_state=42)

    d = range(0, 25)
    gauss_test_acc = []
    PCA_gauss_test_acc = []
    pois_test_acc = []
    gc_pois_test_acc = []
    sel_pois_test_acc_ftest = []
    sel_pois_test_acc_chi= []
    chance = []

    for i in d:
        gauss_test_temp = []
        PCA_gauss_test_temp = []
        pois_test_temp = []
        gc_pois_test_temp = []
        sel_pois_test_temp_ftest = []
        sel_pois_test_temp_chi= []

        for train_index, test_index in skf:
            y_train, y_test = y[train_index], y[test_index]
            X_train, X_test = X[train_index], X[test_index]
            Xb_train, Xb_test = Xb[train_index], Xb[test_index]
            Xc_train, Xc_test = Xc[train_index], Xc[test_index]
            Xd_train, Xd_test = Xd[train_index], Xd[test_index]
            Xe_train, Xe_test = Xe[train_index], Xe[test_index]

            # Gaussian
            learner = GaussianNB().fit(X_train, y_train)
            gauss_test_temp.append(laxScore(y_test, learner.predict(X_test), i))

            # PCA reduced Gaussian
            learner = GaussianNB().fit(Xb_train, y_train)
            PCA_gauss_test_temp.append(laxScore(y_test, learner.predict(Xb_test), i))

            # Poisson
            learner = PoissonNB().fit(X_train, y_train)
            pois_test_temp.append(laxScore(y_test, learner.predict(X_test), i))

            # GoodCell Poisson
            learner = PoissonNB().fit(Xc_train, y_train)
            gc_pois_test_temp.append(laxScore(y_test, learner.predict(Xc_test), i))

            # Univariate reduced Poisson Chi2 test
            learner = PoissonNB().fit(Xd_train, y_train)
            sel_pois_test_temp_chi.append(laxScore(y_test, learner.predict(Xd_test), i))

            # Univariate reduced Poisson F test
            learner = PoissonNB().fit(Xe_train, y_train)
            sel_pois_test_temp_ftest.append(laxScore(y_test, learner.predict(Xe_test), i))
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
        gauss_test_acc.append(np.mean(np.asarray(gauss_test_temp)))
        PCA_gauss_test_acc.append(np.mean(np.asarray(PCA_gauss_test_temp)))
        pois_test_acc.append(np.mean(np.asarray(pois_test_temp)))
        gc_pois_test_acc.append(np.mean(np.asarray(gc_pois_test_temp)))
        sel_pois_test_acc_chi.append(np.mean(np.asarray(sel_pois_test_temp_chi)))
        sel_pois_test_acc_ftest.append(np.mean(np.asarray(sel_pois_test_temp_ftest)))


    plt.plot(d, gauss_test_acc)
    plt.plot(d, PCA_gauss_test_acc)
    plt.plot(d, pois_test_acc)
    plt.plot(d, gc_pois_test_acc)
    plt.plot(d, sel_pois_test_acc_chi)
    plt.plot(d, sel_pois_test_acc_ftest)
    plt.plot(d, chance)
    plt.ylabel('Accuracy')
    plt.xlabel('Distance')
    plt.legend(['Gaussian','PCA Reduced Gaussion', 'Poisson', 'Heuristic Reduced Poisson','Univariate Reduced Poisson (Chi2 test, 15 features)','Univariate Reduced Poisson (F test)','Chance'], loc='lower right')
    plt.show()
    #    plt.savefig('n228/Fr1AccDimRed.png')

def permutation():
    file = 'data/n228_bcdefgh.mat'
    dat = data.load(file)
    X, y = data.build(dat, range(0, 96), 'fr1', 17)

    # Univariate Feature Selection
    select = SelectKBest(f_classif,k=27).fit(X,y)
    Xa = select.transform(X)

    # Select good cell with heuristic
    channel = data.goodCell(dat)
    Xb, y = data.build(dat, channel, 'fr1', 17)

    # PCA Dimentionnality Reduction
    pca = PCA(n_components=38)
    Xc = pca.fit_transform(X)


    dat = [X, Xa, Xb, X, Xc,Xa]
    pNB = PoissonNB()
    gNB = GaussianNB()
    classifiers = [pNB,pNB,pNB,gNB,gNB,gNB]
    label = ['Poisson Unreduced', 'Poisson Univariate Reduction', 'Poisson Heuristic Reduction', 'Gaussion No reduction', 'Gaussian PCA reduction', 'Gaussian Univariate Reduction']
    scores = []
    perm_scores = []
    p_value = []

    for i in range(0,len(dat)):
        score, permutation_score, pvalue = permutation_test_score(classifiers[i], dat[i], y, cv=StratifiedKFold(y, n_folds=3, shuffle=True, random_state=42),n_permutations=100, n_jobs=-1, random_state=42, scoring=make_scorer(error_distance, greater_is_better=False))
        scores.append(score)
        perm_scores.append(np.mean(permutation_score))
        p_value.append(pvalue)

    ind = np.arange(len(scores))
    plt.bar(ind, scores)
#    ax.set_xticks(ind)
#    ax.set_xticklabels(label)
    plt.plot(ind, perm_scores)


    plt.show()


    print "Average Distance between real location and predicted location"
    print score
    print "Chance Performance, from permutation"
    print np.mean(permutation_score)
    print "p-value"
    print pvalue

def localization():
    file = 'data/n191_bcde.mat'
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
    permutation()

