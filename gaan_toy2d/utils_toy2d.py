import numpy as np
import itertools
import matplotlib.pyplot as plt

testcase = 'SQUARE'
BATCH_SIZE = 2000
# Shape 1: Square dots
means1 = [np.array([i, j]) for i, j in itertools.product(range(-4, 5, 2),
                                                           range(-4, 5, 2))]

# Shape 2: Plus dots
means2 = [[-3,  0],
 [-2,  0],
 [-1,  0],
 [ 0,  0],
 [ 1,  0],
 [ 2,  0],
 [ 3,  0],
 [ 0, -3],
 [ 0, -2],
 [ 0, -1],
 [ 0,  1],
 [ 0,  2],
 [ 0,  3]]

# Shape 3: Sine dots
a = range(-180, 15, 30)
a = np.asarray(a)
b = range(0,210,30)
b = np.asarray(b)
x = np.cos(a/180*np.pi)
x = np.concatenate(([x[1:]-1,1-x]))
y = np.sin(b/180*np.pi)
y = np.concatenate(([y[1:],0-y]))
print(a,b)
means3 = []
for i in range(len(x)-1):
    means3.append([x[i],y[i]])

if testcase == 'SQUARE':
    logfile = 'toydata.txt'
    means = means1
elif testcase == 'PLUS':
    logfile = 'plusdata.txt'
    means = means2
elif testcase == 'SINE':
    logfile = 'sinedata.txt'
    means = means3
variances = [0.05 ** 2 * np.eye(len(mean)) for mean in means]
file = open(logfile,"w")

def maxabs(a, axis=None):
    amax = a.max(axis)
    amin = a.min(axis)
    return np.where(-amin > amax, amin, amax)

def lim(a, axis=None):
    amax = a.max(axis)
    amin = a.min(axis)
    avg = (amax-amin)/2
    start = amin - avg
    stop = amax + avg
    return [start,stop]

def sample_z(mean,variance,size):
    return np.random.multivariate_normal(mean= mean, cov= variance, size=size)

def plot_data(data,save_path = 'result', plotcnt = 0, title = 'X_sample'):
    nbpoint = len(data)
    plt.figure(figsize=(5,5))
    plt.scatter(data[:nbpoint,0], data[:nbpoint,1], c='k', s=0.05)
    axes = plt.gca()
    axes.set_xlim(lim(data[:,0]))
    axes.set_ylim(lim(data[:,1]))
    plt.title(title)
    plt.savefig(save_path + str(plotcnt) + '.png',  bbox_inches='tight')
    plt.close()

def read_toydata(toyfile):
    fid = open(toyfile,'r')
    lines = fid.readlines()
    data = []
    for line in lines:
        line = line.replace('[', '')
        line = line.replace(']', '')
        data.append([float(curr_num) for curr_num in line.split()])
	fid.close()
    return np.array(data)

def normalize_toydata(toydata, testcase, var):
    centroids = callback_centroids(testcase)
    centroids = (centroids/maxabs(np.float32(toydata))+1)/2
    var = (var/maxabs(toydata))/np.sqrt(2)
    toydata = (toydata/maxabs(toydata)+1)/2
    toydata_size = len(toydata)
    #print('Len of all data %d'%(len(toydata)))
    return toydata, toydata_size, centroids ,var 
    
def evaluate_toydata(data,centroids,vars):
    inclass = [[len(data)]]
    for i in range(len(centroids)):
        subdata = data - centroids[i]
        distance = np.linalg.norm(subdata,axis=1)
        nb = (distance<=vars).sum()
        inclass.extend([nb])
    return inclass

def callback_centroids(tc = 'SINE'):
    if tc == 'SQUARE':
        return means1
    elif tc == 'PLUS':
        return means2
    elif tc == 'SINE':
        return means3
    else:
        print("Wrong Input")

def random_batches(dataset, batch_size):
    data_size = len(dataset)
    idx = np.random.permutation(data_size)
    dataset = dataset[idx,:]
    return dataset[0:batch_size,:]
