import matplotlib.pyplot as plt
import pickle
import numpy as np
from scipy.interpolate import make_interp_spline, BSpline, spline

if __name__ == '__main__':
    env_name='FetchReach-v1'
    save_path = "./save/"+env_name
    data=None
    with open(f'{save_path}/metrics.dat','rb') as f:
        data = pickle.load(f)

    u=data.shape[0]
    data=data.reshape((u//2,2))
    #print(data)

    rewards=data[:,0]
    losses=data[:,1]
    x=np.arange(0,u//2,1)
    x_smooth=np.linspace(x.min(),x.max(),200)
    l_smooth=spline(x,losses,x_smooth)
    r_smooth=spline(x,rewards,x_smooth)

    plt.subplot(2,1,1)
    plt.plot(x_smooth,l_smooth)
    plt.title('Loss')
    plt.subplot(2,1,2)
    plt.plot(x_smooth,r_smooth)
    plt.title('Rewards')
    plt.show()
