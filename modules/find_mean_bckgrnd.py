import os
op = os.path
opb, opj = op.basename, op.join
import cv2
import numpy as np
from matplotlib import pyplot as plt

class MEAN_BACKGROUND():
    '''
    Find the background using the average picture
    '''
    def __init__(self, list_addr_imgs):
        '''
        '''
        self.N = len(list_addr_imgs)
        self.limg = []
        for addr_img in list_addr_imgs:
            self.limg += [cv2.imread(addr_img)]

    def find_shift(self, img0, img1, debug=[]):
        '''
        Find the shift in pixels between img0 and img1
        '''
        f0 = cv2.cvtColor(np.float32(img1), cv2.COLOR_BGR2GRAY)
        f1 = cv2.cvtColor(np.float32(img0), cv2.COLOR_BGR2GRAY)
        shift = cv2.phaseCorrelate(f1, f0)[0]
        if 0 in debug:
            print(f'shift is {shift}')
            print(f'img0.shape is { img0.shape }')

        return shift

    def shift_img(self, img, shift):
        '''
        Shift img with shift translation
        '''
        M = np.float32([[1, 0, shift[0]], [0, 1, shift[1]]])
        img_shifted = cv2.warpAffine(img.copy(), M, (img.shape[1], img.shape[0]))

        return img_shifted

    def align_img0_on_img1(self, img0, img1):
        '''
        Align img0 on img1
        '''
        shift = self.find_shift(img0, img1)
        img0_shifted = self.shift_img(img0, shift)

        return img0_shifted

    def shift_and_add(self, img0, img1, ratio=0.5):
        '''
        Shift img0 on img1 and add them
        '''
        shifted_img0 = self.align_img0_on_img1(img0, img1)
        added = cv2.addWeighted(shifted_img0, ratio, img1, 1-ratio, 0.0)

        return added

    def shift_and_remove(self, img0, img1):
        '''
        Shift img0 on img1 and substract them
        '''
        shifted_img0 = self.align_img0_on_img1(img0, img1)
        sub = img0-img1

        return sub

    def extract_bckgrnd(self, ind_range=[0,10],
                        save_all=False,
                        show_last=True,
                        show_ith=None,
                        debug=[0]):
        '''
        Find the non moving part of the images
        '''
        last_ind = ind_range[1]-1
        # first image
        interm = self.shift_and_add(self.limg[0], self.limg[1], ratio=0.5)
        # mean with following images
        if 0 in debug:
            print('In extract_bckgrnd...')
        for i in range(ind_range[0],ind_range[1]):

            self.interm = self.shift_and_add(self.limg[i], interm, ratio=1/(i+2))
            self.pic_name = f'mean with {i-ind_range[0]} pictures.tiff'
            if show_ith != None:
                if i == show_ith:
                    print(f'keep image {i}')
                    self.pic_ith = self.interm
            if save_all:
                # save the whole evolution of the mean
                cv2.imwrite(self.pic_name, self.interm)
            else:
                # save only the last picture for the mean
                if i == last_ind:
                    cv2.imwrite(self.pic_name, self.interm)
                    if show_last:
                        plt.title(f'pic{i}')
                        plt.imshow(self.interm)
                    if show_ith != None:
                        try:
                            print(f'showing pic {show_ith}')
                            plt.figure()
                            plt.title(f'pic{show_ith}')
                            plt.imshow(self.pic_ith)
                        except:
                            print(f'Cannot plot image {show_ith}')
