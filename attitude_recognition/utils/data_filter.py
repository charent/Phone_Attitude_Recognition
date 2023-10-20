from re import S
import numpy as np
from pykalman import KalmanFilter
from tqdm.std import trange

class Kalman_Filter():
    def __init__(self, damping: int=1):
        
        self.kf = KalmanFilter(
            initial_state_mean=None,
            initial_state_covariance=damping,
            observation_covariance=1,
            transition_covariance=1,
            transition_matrices=1
        )

    def do_smooth(self, data: np.ndarray):
        '''
        data: [[...],[...],...]
        '''
        column = data.shape[1]

        ret = []
        for i in range(column):
            pred_state, state_conv = self.kf.smooth(data[:, i])
            pred_state = np.squeeze(pred_state, axis=1)
         
            ret.append(pred_state)

        ret = np.array(ret)
        ret = np.transpose(ret)

        return ret


if __name__ == '__main__':
    from scipy.signal import savgol_filter

    a = [1,2,3,4,5,7,8,11,2]
    print(len(a))
    s = savgol_filter(a, 5, 3)
    print(s)
    