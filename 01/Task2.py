import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab

if __name__ == "__main__":
    data = np.loadtxt('data/whData.dat',dtype=np.object,comments='#',delimiter=None)
    X = np.array(data[:,1].astype(np.float))

    mean = np.mean(X)
    std = np.std(X)

    fig = plt.figure()
    axs = fig.add_subplot(111)

    x = np.linspace(mean-3.5*std,mean+3.5*std,500)
    axs.plot(x,mlab.normpdf(x,mean,std), 'b', label = 'normal')

    X = np.column_stack((X,np.zeros(X.shape[0])))

    axs.plot(X[:,0], X[:,1], marker='o', c='b', label = 'data')

    leg = axs.legend(loc='upper left', shadow=True, fancybox=True, numpoints=1)
    leg.get_frame().set_alpha(0.5)

    plt.show()
    plt.savefig("out/Task2.pdf", facecolor='w', edgecolor='w',
                    papertype=None, format='pdf', transparent=False,
                    bbox_inches='tight', pad_inches=0.1)
    plt.close()
