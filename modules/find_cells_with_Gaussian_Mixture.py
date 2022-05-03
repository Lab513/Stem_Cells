from sklearn.mixture import GaussianMixture
from matplotlib import pyplot as plt
import os
import cv2
import numpy as np
op = os.path
opb, opd, opj = op.basename, op.dirname, op.join

class FIND_CLUSTERS_WITH_GM():
    '''
    '''
    def __init__(self):
        '''
        '''
        self.img_size = 512

    def plot_distrib(self, arr_pts):
        '''
        '''
        fig, ax = plt.subplots()
        plt.xlim(0,self.img_size)
        plt.ylim(0,self.img_size)
        for p in arr_pts:
            ax.plot(p[0], p[1], 'kx')
        plt.show()

    def control_nb_clusters_effect(self, nb_tests):
        '''
        Plot the different scores according to the number of clusters seeked
        '''
        plt.figure()
        plt.title('diffp')
        plt.plot(range(1,self.nb_tests),self.diffp)
        ###
        plt.figure()
        plt.title('cov ratio')
        plt.plot(range(1,self.nb_tests),self.cov_score)
        plt.show()

    def find_optim_gm(self, arr_pts, show_plot=False):
        '''
        Find the Gaussian Mixture given by the optimal number of clusters
        '''
        diffp = []
        self.cov_score = []
        self.nb_tests = 6
        for i in range(1,self.nb_tests):
            gm = GaussianMixture(n_components=i, random_state=0).fit(arr_pts)
            p = gm.means_[0]
            cov = gm.covariances_[0]
            #plt.plot(p[0], p[1], 'o')
            try:
                self.diffp += [norm(p-pold)]
            except:
                print('pold not existing')
            self.cov_score += [abs(cov[0][0]/cov[1][0])]
            pold = p
        self.opt_cov = self.cov_score.index(max(self.cov_score[1:]))+1
        if show_plot:
            self.control_nb_clusters_effect()

    def draw_clusters_contours(self, addr_fig_pos, debug=[0]):
        '''
        Contours for the detections supposed to be
        clouds of points with typical variance and shapes.. 
        '''
        img = cv2.imread(addr_fig_pos)
        img = img.astype('float32')
        img = img/255
        img = cv2.flip(img, 0)
        img = np.expand_dims(img, 0)
        pred_clusters = self.mod_cluster_pos.predict(img)
        pred_img_clusters = pred_clusters[0]*255
        pred_img_clusters = pred_img_clusters.astype('uint8')
        plt.imshow(pred_img_clusters)
        # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, thr = cv2.threshold(pred_img_clusters, 127, 255, 0)      # Threshold
        thr = thr.astype('uint8')
        cntrs_clusters, _ = cv2.findContours(thr, cv2.RETR_TREE,
                                    cv2.CHAIN_APPROX_SIMPLE)[-2:]
        img_with_clust = np.squeeze(self.img)
        img_with_clust = cv2.flip(img_with_clust, 0)
        if 0 in debug:
            print(f'img_with_clust.shape { img_with_clust.shape }')
            print(f'img_with_clust.min() { img_with_clust.min() }')
            print(f'img_with_clust.max() { img_with_clust.max() }')
        cv2.drawContours(img_with_clust, cntrs_clusters, -1, (0, 255, 255), 1)
        plt.imshow(img_with_clust)
        plt.savefig( opj(self.folder_results,
                         f'found clusters for well{self.well}.png') )


    def plot_pts_distrib(self, arr_pts):
        '''
        Plot distribution of the positions for ML
        '''
        nbpix = 512
        fig, ax = plt.subplots()
        fig = plt.figure(figsize=(5.12, 5.12), dpi=100)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        ax.set_xlim(0,nbpix)
        ax.set_ylim(0,nbpix)
        for pt in arr_pts:
            plt.plot(pt[0], nbpix-pt[1], 'kx')
        addr_fig_pos = opj(self.folder_results,
                           f'all the positions for well {self.well}'\
                           ' for false pos detect.png')

        plt.savefig(addr_fig_pos)
        print('Find the clusters from the detections')
        self.draw_clusters_contours(addr_fig_pos)

    def plot_optim_GM(self, arr_pts):
        '''
        Plot the detections found on the final image..
        '''
        print('plot optimal Gaussian Mixture...')
        #opt = diffp.index(min(diffp))+1
        print(f'current well is {self.well}')

        print(f'opt is {self.opt_cov}')
        gm = GaussianMixture(n_components=self.opt_cov, random_state=0).fit(arr_pts)
        pm = gm.means_[0]
        cov = gm.covariances_[0]
        print(f'pm {pm}')
        print(f'cov {cov}')
        ####
        #fig, ax = plt.subplots()
        plt.figure()
        # using the last picture
        plt.imshow(np.squeeze(self.img))
        # plot a cross on the detections positions..
        for pt in arr_pts:
            plt.plot(pt[0], pt[1], 'kx')
        plt.plot(pm[0], pm[1], 'ro')
        addr_fig_pos = opj(self.folder_results,
                           f'all the positions for well {self.well}'\
                           ' for false pos detect superp.png')

        plt.savefig(addr_fig_pos)
