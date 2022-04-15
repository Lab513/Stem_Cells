import cv2
import numpy as np
from sklearn.mixture import GaussianMixture
from matplotlib import pyplot as plt

class FIND_CLUSTERS_WITH_GM():
    '''
    '''
    def __init__(self):
        '''
        '''

    def plot_distrib(self, arr_pts):
        '''
        '''
        fig, ax = plt.subplots()
        plt.xlim(0,512)
        plt.ylim(0,512)
        for p in arr_pts:
            ax.plot(p[0], p[1], 'kx')
        plt.show()

    def find_optim_gm(self, arr_pts, show_plot=False):
        '''
        '''
        diffp = []
        self.cov_score = []
        for i in range(1,6):
            gm = GaussianMixture(n_components=i, random_state=0).fit(arr_pts)
            p = gm.means_[0]
            cov = gm.covariances_[0]
            #plt.plot(p[0], p[1], 'o')
            try:
                diffp += [norm(p-pold)]
            except:
                print('pold not existing')
            self.cov_score+=[abs(cov[0][0]/cov[1][0])]
            pold = p
        self.opt_cov = self.cov_score.index(max(self.cov_score[1:]))+1
        if show_plot:
            plt.figure()
            plt.title('diffp')
            plt.plot(range(1,6),diffp)
            ###
            plt.figure()
            plt.title('cov ratio')
            plt.plot(range(1,6),self.cov_score)
            plt.show()

    def plot_optim_GM(self, arr_pts, ax):
        '''
        Plot the circle of variance for the optimal cluster denumbering..
        '''
        #opt = diffp.index(min(diffp))+1
        print(f'current well is {well}')

        print(f'opt is {self.opt_cov}')
        gm = GaussianMixture(n_components=self.opt_cov, random_state=0).fit(arr_pts)
        pm = gm.means_[0]
        cov = gm.covariances_[0]
        print(f'pm {pm}')
        print(f'cov {cov}')
        ####
        # fig, ax = plt.subplots()
        # plt.xlim(0,512)
        # plt.ylim(0,512)
        # plt.title(f'dispersion for {well}')
        for pt in arr_pts:
            ax.plot(pt[0], pt[1], 'kx')
        ax.plot(pm[0], pm[1], 'ro')
        circ = plt.Circle((pm[0],pm[1]), cov[0][0]/6, color='b', fill=False)
        ax.add_patch(circ)
        #plt.show()
