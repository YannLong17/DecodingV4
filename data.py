import numpy as np
import scipy.io as sio
from matplotlib import pyplot as plt


def load(file):
    dat = sio.loadmat(file,squeeze_me=False)
    return dat

def getSacc(dat):
    X1 = dat['makeStim']['fixX'][0, 0][0][0]
    X2 = dat['makeStim']['fixX2'][0, 0][0][0]
    Y1 = dat['makeStim']['fixY'][0, 0][0][0]
    Y2 = dat['makeStim']['fixY2'][0, 0][0][0]
    sacc = [X2 - X1, Y2 - Y1]
    return sacc

def goodCell(dat):
    from scipy.ndimage.measurements import center_of_mass as CM
    fr1 = dat['fr1'][:100,0]
    fr3 = dat['fr3'][:100,0]
    sacc = getSacc(dat)
    time = 17

    channel = list()
    for i in range(0,96):

        # Get Center of Mass
        fix1 = dat['CofM'][0, i]['fix1'][0, 0][:, time]
        fix2 = dat['CofM'][0, i]['fix2'][0, 0][:, time]

        #Compare Center of Mass shift to the Saccade Vector:
        if compare(fix2-fix1, sacc):
            channel.append(i)

    return channel

def compare(fix, sacc):
    maxAngle = 1
    minDisplacement = np.linalg.norm(sacc)/2
    bool=1
    if np.any(angle(fix, sacc) > maxAngle or np.linalg.norm(fix) < minDisplacement or np.isnan(fix)):
        bool=0

    return bool

def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

def angle(v1, v2):
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

def build(dat,channel,condition,time):
    col = len(channel) + 1
    data = np.zeros((1,col))
    data[0,:-1]=channel
    data[0,-1]=-1
    for i in range(0,100):
        row = len(dat[condition][i, 0][0, :, time])
        temp = np.zeros((row, col))
        k = 0
        for j in channel:
            temp[:, k] = dat[condition][i, 0][j, :, time]
            k += 1
        location = np.empty(row)
        location[...] = i
        temp[:, -1] = location
        data = np.vstack((data,temp))

    X = data[1:, :-1]
    Y = data[1:, -1]
    return X,Y

def visualize():
    file='data/n228_bcdefgh.mat'
    dat=load(file)
    fr1 = dat['fr1'][:100,0]
    fr3 = dat['fr3'][:100,0]
    array1 = np.zeros(100)
    array2 = np.zeros(100)
    time = 17

    from sklearn.feature_selection import SelectKBest, chi2
    # Univariate selection based on Fr1
    X, y = build(dat,range(0,96),'fr1',time)
    selected = SelectKBest(chi2,k=38).fit(X,y).get_support(indices = True)

    # Heuristic Selection based center of mass Shift
    good = goodCell(dat)

    for i in range(0,96):
        for j in range(0,100):
            array1[j] = np.mean(fr1[j,][i,:,time])
            array2[j] = np.mean(fr3[j,][i,:,time])

        plt.subplot(2, 1, 1)
        plt.imshow(np.reshape(array1,[10,10]).T)
        plt.title(i+1)
        plt.subplot(2, 1, 2)
        plt.imshow(np.reshape(array2,[10,10]).T)

        if i in good and i in selected :
            plt.title("Good in Both")
        elif i in good and i not in selected:
            plt.title("Heuristic Only")
        elif i in selected and i not in good:
            plt.title("Univariate Selection Only")
        else:
            plt.title("Bad")

        plt.savefig('Figures/n228channel{0}.png'.format(i+1))

if __name__ == '__main__':
    visualize()