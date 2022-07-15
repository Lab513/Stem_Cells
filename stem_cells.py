'''
Program for segmenting and counting the stem cells
'''

from modules.find_mean_bckgrnd import MEAN_BACKGROUND as MB
from modules.gap_statistics import optimalK
from modules.find_cells_with_Gaussian_Mixture import FIND_CLUSTERS_WITH_GM as FGM
from datetime import datetime
from time import time
from pathlib import Path
from matplotlib import pyplot as plt
from tensorflow.keras import models
from sklearn.metrics import pairwise_distances_argmin
from scipy.linalg import norm
from modules.BC import correctbaseline

import pandas as pd
import glob
import re
import os
import sys
import numpy as np

import yaml
import shutil as sh
import pandas as pd
import cv2
import pickle as pk
op = os.path
opb, opd, opj = op.basename, op.dirname, op.join


class Logger(object):
    '''
    Logger for mda experiment
    '''
    def __init__(self, folder_results):
        self.terminal = sys.stdout
        self.log = open(opj(folder_results, 'log.dat'), "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def __getattr__(self, attr):
        return getattr(self.terminal, attr)

    def flush(self):
        pass


class STEM_CELLS(FGM):
    '''
    Detect and count the stem cells
    The program uses Tensorflow
    '''

    def __init__(self, addr_folder,
                 list_models=['stem_ep30'],
                 model_area='stem_area_ep5',
                 model_clustering='cluster_pos_ep10',
                 manual_annotations=r"manual_annot/AD63_manual.xlsx",
                 min_area=2,
                 cell_type='HSC',
                 cmp_thr=200,
                 span=None):
        '''
        addr_folder: address of the images
        model : model shortcut used for counting the cells
        '''
        FGM.__init__(self) # find clusters
        # , engine= 'openpyxl'
        if manual_annotations:
            xls = pd.ExcelFile( fr"manual_annot/{manual_annotations}" )
            self.manual_df = xls.parse(0)
            self.hsc = self.manual_df.loc[self.manual_df['Cell_type'] == cell_type]
        ##
        # font size
        self.fsize = 60
        # composite threshold
        self.cmp_thr = cmp_thr
        self.list_models = list_models
        if len(list_models) > 1:
            self.curr_model = 'multi'
        else:
            self.curr_model = list_models[0]
        self.addr_folder = addr_folder                 # path for data
        self.id_exp = opb(opd(self.addr_folder))
        self.kind_exp = opb(self.addr_folder)
        self.folder_results = f'results_mod_{self.curr_model}'\
                              f'_data_{self.id_exp}-{self.kind_exp}'\
                              f'_{self.date()}'
        # suffix for spanning range..
        if span:
            suffix_name_proc = self.make_span_suffix(span)
            self.folder_results += f'_{suffix_name_proc}'
        curr_list_mod = []
        with open('models.yaml') as f:
            dic_models = yaml.load(f, Loader=yaml.FullLoader)
            # full name of the current stems model
            for model in list_models:
                curr_list_mod += [dic_models[model]]
            # full name of the current stem model area
            curr_mod_area = dic_models[model_area]
            curr_mod_cluster = dic_models[model_clustering]

        addr_area_model = Path('models') / curr_mod_area
        addr_cluster_model = Path('models') / curr_mod_cluster

        print(f'addr_area_model is {addr_area_model}')
        print(f'addr_cluster_model is {addr_cluster_model}')
        ####
        self.list_mod_stem = []
        # load the models for stem cells detection
        for curr_mod in curr_list_mod:
            addr_model = Path('models') / curr_mod
            print(f'addr_model is {addr_model}')
            self.list_mod_stem += [ models.load_model(addr_model) ]
        # load the model for finding the area where are the stem cells
        self.mod_stem_area = models.load_model(addr_area_model)
        self.mod_cluster_pos = models.load_model(addr_cluster_model)
        self.prepare_result_folder()        # create the folder for the results
        self.min_area = min_area            # minimal area for cell
        self.max_area = 50                  # maximal area for cell
        self.lev_thr = 100                  # threshold level for segmentation
        self.dic_nbcells = {}
        # tnbc : time and nb of cells
        self.dic_tnbc = {'direct_ML':{'well':[],
                         'time':[],
                         'nbcells':[]},
                         'stat':{'well':[],
                         'time':[],
                         'nbcells':[]}}     # dictionary of time and nb cells
        self.list_tnbc = []                 # list of time and nb cells

    def make_span_suffix(self, span):
        '''
        Suffix for the span range in the folder name..
        '''
        lspan, nspan = span
        if len(lspan) > 1:
            spansuff = f'{lspan[0]}-{lspan[-1]}_{nspan}'
        else:
            spansuff = f'{lspan[0]}_{nspan}'

        return spansuff

    def retrieve_times_nb_cells(self, well, debug=[]):
        '''
        Retrieving the time and number of cells from the Excel file.
        '''
        if 0 in debug:
            print(f'Retrieving manual annotations for well {well}')
        w = self.hsc.loc[ self.hsc['Well'] == well]
        ldiv_dates = [0]
        lnb_cells = [0]
        nb_cells = 0
        ltimes = ['T1','T2.1','T2.2','T3.1','T3.2','T3.3','T3.4']
        for t in ltimes:
            try:
                div_date = w[t].values[0]
                if 1 in debug:
                    print(f't is {t}')
                    print(f'div_date is {div_date}')
                ldiv_dates += [div_date, div_date]
                nb_cells = ltimes.index(t)+2
                lnb_cells += [lnb_cells[-1], nb_cells]
            except:
                print('this time does not exist')

        return ldiv_dates, lnb_cells


    def date(self):
        '''
        Return a string with day, month, year, Hour and Minute..
        '''
        now = datetime.now()
        dt_string = now.strftime("%d-%m-%Y-%H-%M")
        return dt_string

    def last_two_levels(self, addr):
        '''
        From /a/b/c/d returns c/d
        '''
        return opj(opb(opd(addr)), opb(addr))

    def prepare_result_folder(self):
        '''
        Create the folder "results"
        '''
        os.mkdir(self.folder_results)                              # results folder
        sys.stdout = Logger(self.folder_results)
        with open(f'{self.folder_results}/proc_infos.yaml', 'w') as f_w:
            # save model name in proc_infos.yaml
            dic_infos = {'dataset': self.last_two_levels(self.addr_folder),
                         'models': self.list_models}
            yaml.dump(dic_infos, f_w)

    def mdh_to_nb(self, l, make_lmdh=True):
        '''
        from month-day-hour format to
         the number mdh for sorting the dates.
        02m12d03h --> 21203
        '''
        mdh = re.findall(r'\d+m\d+d_\d+h', l)[0]
        if mdh[0] == '0':
            mdh = mdh[1:]
        if make_lmdh:
            self.lmdh += [mdh]
        nb = int(mdh.replace('d_', '')
                    .replace('h', '')
                    .replace('m', ''))
        return nb

    def find_time_interval(self):
        '''
        Find the interval of time between the acquisitions.
        '''
        n1 = self.mdh_to_nb(self.lmdh[1], make_lmdh=False)
        n0 = self.mdh_to_nb(self.lmdh[0], make_lmdh=False)
        self.delta_exp = n1-n0
        print(f'Time interval between images is {self.delta_exp} hours')
        if self.delta_exp > 10:
            print('There is an issue with the time interval')

    def list_imgs(self, well=None, debug=[]):
        '''
        List of the images for a given well
        '''
        self.well = well
        glob_string = f'{self.addr_folder}/*_{well}_*.tif'
        if 0 in debug:
            print(f'glob_string is {glob_string}')
        self.addr_files = glob.glob(glob_string)
        if 1 in debug:
            print(f'self.addr_files is {self.addr_files}')
        print(f'Nb of images to process is {len(self.addr_files)}')
        # sort the files names with date
        self.addr_files.sort(key=lambda x: self.mdh_to_nb(x))
        # list month day hour
        self.lmdh.sort(key=lambda x: self.mdh_to_nb(x, make_lmdh=False))
        self.find_time_interval()
        self.mb = MB(self.addr_files)

    def prep_img(self, addr_img):
        '''
        Prepare image for unet
        '''
        img = cv2.imread(addr_img)
        img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_AREA)
        self.img_orig = img.copy()
        img = img.astype('float32')
        img = img/255
        img = np.expand_dims(img, 0)

        return img

    def insert_text(self, img, txt, pos=(10, 10)):
        '''
        Insert text in the image
        '''
        font = cv2.FONT_HERSHEY_SIMPLEX              # font
        fontScale = 0.8
        color = (255, 255, 255)                      # RGB when inserted
        thickness = 2                                # Line thickness of 2 px
        img = cv2.putText(img, txt, pos, font,
                          fontScale, color, thickness, cv2.LINE_AA)
        return img

    def make_mask_from_contour(self,cnt,debug=[]):
        '''
        '''
        if 0 in debug:
            print(f'self.img.shape = {self.img.shape}')
        _, h, w, _ = self.img.shape
        mask = np.zeros((h, w), np.uint8)
        cv2.drawContours(mask, [cnt], -1, (255, 255, 255), -1) # fill contour

        return mask

    def IoU_filter(self,c1, c2, debug=[]):
        '''
        Compare Intersection over Union..
        '''
        mask0 = self.make_mask_from_contour(c1)           # previous contour
        mask1 = self.make_mask_from_contour(c2)           # new contour..
        inter = np.logical_and(mask0, mask1)
        union = np.logical_or(mask0, mask1)
        iou_score = np.sum(inter) / np.sum(union)
        if 0 in debug:
            print(f"### IoU score for  is {iou_score}")
        return iou_score

    def find_cntrs(self, img,
                   thresh=None,
                   filt_surface=False,
                   debug=[1]):
        '''
        Find the contours in the thresholded prediction
        '''
        try:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        except:
            gray = img
        if not thresh:
            ret, thr = cv2.threshold(gray, self.lev_thr, 255, 0)      # Threshold
        else:
            ret, thr = cv2.threshold(gray, thresh, 255, 0)      # Threshold
        thr = thr.astype('uint8')
        cntrs, _ = cv2.findContours(thr, cv2.RETR_TREE,
                                    cv2.CHAIN_APPROX_SIMPLE)[-2:]
        if 1 in debug:
            print(f'Before filtering, len(cntrs) = {len(cntrs)}')
        if len(cntrs) > 7:
            for c in cntrs:
                if 2 in debug:
                    print(f'cv2.contourArea(c) is {cv2.contourArea(c)}')
        # filter on area
        if filt_surface:
            filt_cntrs = [c for c in cntrs if self.max_area
                          > cv2.contourArea(c) > self.min_area]
            cntrs = filt_cntrs

        if 1 in debug:
            print(f'After filtering, len(cntrs) = {len(cntrs)}')

        return cntrs

    def correct_up(self, nbcells):
        '''
        '''
        nnbcells = []
        for i,c in enumerate(nbcells):
            if i >1:
                if c-nbcells[i-1]< 0:
                    nnbcells += [nbcells[i-1]]
                else:
                    nnbcells += [c]

        nnbcells = np.array(nnbcells)

        return nnbcells

    def increasing_values(self, debug=[]):
        '''
        filter for having increasing values only
        '''
        print('filter for increasing values.. ')
        print(f'initial list is {self.l_level}')
        for i,curr in enumerate(self.l_level):
            if i > 0:
                prev = self.l_level[i-1]
                if prev > curr:
                    if 1 in debug:
                        print(f'change level from {curr} to {prev}')
                    self.l_level[i] = prev

        print(f'self.l_level = {self.l_level}')

    def max_density_levels(self, nbcells,
                                 drange=5,
                                 iterat=8,
                                 nblevels = 10,
                                 debug=[]):
        '''
        Find the levels using the density
        drange : density range is the intervall on which is calculated the density
        '''
        print('make max_density_levels')
        self.l_level = []
        for j,nb in enumerate(nbcells):
            sub = nbcells[j:j + drange]
            if 1 in debug:
                print(f'sub = {sub}')
            maxi = 0
            maxind = 0
            for lev in range(0,nblevels+1):
                ll = np.where(sub == lev)[0]
                if 0 in debug:
                    print(f'lev is {lev}')
                    print(f'll is {ll}')
                # number of occurences
                nbocc = len(ll)
                if nbocc > maxi:
                    # value corresponding to maximum of occurrences
                    maxind = lev
                    maxi = nbocc
            self.l_level += [maxind]

        for i in range(iterat):
            self.l_level = self.correct_up(self.l_level)
        beg_zeros = np.zeros(int(drange/2 + 2*iterat))
        self.l_level = np.concatenate((beg_zeros, self.l_level))
        self.increasing_values()

        return self.l_level

    def draw_pred_contours(self, img, cntrs, debug=[0]):
        '''
        '''
        if 0 in debug:
            print(f'img.shape { img.shape }')
            print(f'In draw_pred_contours, len(cntrs) = {len(cntrs)}')
        _, h, w, _ = img.shape
        mask = np.zeros((h, w), np.uint8)
        # fill contour
        cv2.drawContours(mask, cntrs, -1, (255, 255, 255), -1)
        return mask

    def save_pred(self, i, img, cntrs, cntrs_area):
        '''
        '''
        img_cntrs = self.draw_pred_contours(img, cntrs)
        cv2.imwrite(opj(self.pred_folder, f'img{i}_pred.png'), img_cntrs)
        ###
        img_cntrs_area = self.draw_pred_contours(img, cntrs_area)
        cv2.imwrite(opj(self.pred_folder, f'img{i}_pred_area.png'), img_cntrs_area)

    def save_BF(self, i, cntrs):
        '''
        '''
        cv2.imwrite(opj(self.pred_folder, f'img{i}.png'), self.img_orig)

    def save_BF_superp(self, i, cntrs):
        '''
        '''
        img_superp = self.img_orig.copy()
        try:
            cv2.drawContours(img_superp, cntrs, -1, (0, 0, 255), -1)
        except:
            print('Probably no contours')
        cv2.imwrite(opj(self.pred_folder,
                    f'img{i}_superp.png'), img_superp)

    def insert_text(self, img, text, pos=(50, 50)):
        '''
        '''
        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 0.4
        color = (255, 0, 0)
        thickness = 1
        img = cv2.putText(img, text, pos, font,
                           fontScale, color, thickness, cv2.LINE_AA)

        return img

    def save_BF_with_area(self, i, cntrs_area, debug=[]):
        '''
        '''
        img_superp_area = self.img_orig.copy()
        cv2.drawContours(img_superp_area, cntrs_area, -1, (0, 255, 255), 1)
        area = 0
        for c in cntrs_area:
            curr_area = cv2.contourArea(c)
            if 0 in debug:
                print(f'curr_area = {curr_area}')
            if curr_area > 5e3:
                area = curr_area
        txt_area = f'area = {area}'
        img_superp_area = self.insert_text(img_superp_area, txt_area, pos=(50, 50))

        cv2.imwrite(opj(self.pred_folder,
                    f'img{i}_stem_area.png'), img_superp_area)

    def prep_fig(self):
        '''
        '''
        nbpix = 512
        fig, ax = plt.subplots()
        fig = plt.figure(figsize=(5.12, 5.12), dpi=100)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        ax.set_xlim(0,nbpix)
        ax.set_ylim(0,nbpix)

        return ax

    def save_area_with_detections(self, i, cntr_area, cntrs):
        '''
        '''
        img_cross = self.img_orig.copy()
        cv2.drawContours(img_cross, cntr_area, -1, (0, 255, 255), 1)
        addr_fig_pos = opj(self.pred_folder,f'img{i}_detect_inside.png')
        for c in cntrs:
            pt = self.pos_from_cntr(c)
            cv2.drawMarker(img_cross, (pt[0],pt[1]), color=(0,0,0),
                            markerSize=15, markerType=cv2.MARKER_CROSS, thickness=2)
            # cv2.drawMarker(img, (256,256), color=(0,0,0),
            #                 markerSize=15, markerType=cv2.MARKER_CROSS, thickness=1)
        cv2.imwrite(addr_fig_pos, img_cross)

    def save_well_pics(self, i, img, cntrs, cntrs_area):
        '''
        Save images related to a given well
        '''
        # save predictions contours..
        self.save_pred(i, img, cntrs, cntrs_area)
        # save BF
        self.save_BF(i, cntrs)
        # save BF + pred
        self.save_BF_superp(i, cntrs)
        # save BF + area for stems
        self.save_BF_with_area(i, cntrs_area)
        # save contours with save_area_with_detections
        self.save_area_with_detections(i, cntrs_area, cntrs)

    def from_pred_to_imgpred(self, pred, debug=[]):
        '''
        '''
        if 0 in debug:
            print(f'type(pred[0]) is {type(pred[0])}')
            print(f'pred[0].shape) is {pred[0].shape}')
        pred_img = pred[0]*255
        pred_img = pred_img.astype('uint8')

        try:
            img_pred = cv2.cvtColor(pred_img, cv2.COLOR_BGR2RGB)
        except:
            img_pred = pred_img
        return img_pred

    def make_pred(self, i, f, debug=[0,1,2]):
        '''
        Make the predictions for cells, areas etc..
        '''
        if 0 in debug:
            print(f'predictions for well : {self.well}')
        self.curr_ind = i
        # prepare images
        img = self.prep_img(f)
        # prediction with models stem_cells
        list_img_pred = []
        for mod in self.list_mod_stem:
            pred = mod.predict(img)
            img_pred = self.from_pred_to_imgpred(pred)
            list_img_pred += [img_pred]
        if 1 in debug:

            print('all predictions done.. ')
        # prediction with model stem_cells area
        pred_area = self.mod_stem_area.predict(img)
        pred_img_area = pred_area[0]*255
        pred_img_area = pred_img_area.astype('uint8')
        if 2 in debug:
            print('pred for stems area done.. ')

        return img, list_img_pred, pred_img_area

    def save_nb_cells_max_in_filtered(self):
        '''
        Save the number max of cells in the filterd cells..
        '''
        nb_cells_max = max(self.lnbcells_stat)
        file_nb_cntrs = opj(self.pred_folder, f'nb_cells_max.yaml')
        with open(file_nb_cntrs, 'w') as f_w:
            yaml.dump(nb_cells_max, f_w)

    def save_scores(self):
        '''
        Scores for each well
        '''
        print('saving the scores in scores.yaml')
        addr_scores = opj(self.pred_folder, f'scores.yaml')
        with open(addr_scores, 'w') as f_w:
            scores = { 'ml': f'{self.curr_score_ml}',
                       'stat': f'{self.curr_score_stat}' }
            yaml.dump(scores, f_w)

    def extract_date(self, f):
        '''
        '''
        nimg = opb(f)
        ymdh = re.findall(r'\d+y\d+m\d+d_\d+h', nimg)[0]

        return ymdh

    def save_result_in_dict(self, kind, debug=[0]):
        '''
        Save the results in the dict which will be converted to csv
        kind : stat or direct_ML
        '''
        ##
        dic = self.dic_tnbc[kind]
        dic['well'] += [self.well]*len(self.ltimes)
        dic['time'] += self.ltimes
        if 0 in debug:
            print(f'len(self.lnbcells_stat_levels) { len(self.lnbcells_stat_levels) }')
            print(f'len(self.lnbcells_levels) { len(self.lnbcells_levels) }')
        # nbcells for stat method
        if kind == 'stat':
            dic['nbcells'] += list(self.lnbcells_stat_levels)
        # nbcells for direct ML method
        elif kind == 'direct_ML':
            dic['nbcells'] += list(self.lnbcells_levels)

    def find_false_pos_bckgd(self, i, range_bckgd , debug=[0,1]):
        '''
        '''
        try:
            if 0 in debug:
                print('find cntrs for false positive...')
            # find static shapes in the image using average
            self.mb.extract_bckgrnd(ind_range=[i, range_bckgd+i], save_all=True)
            img_bckgd = cv2.imread(self.mb.pic_name)
            _, img_pred_bckgd = self.make_pred(i, self.mb.pic_name)
            addr_false_pos = opj(self.pred_folder, f'false_positive_{i}.png')
            # save false positive picture
            cv2.imwrite(addr_false_pos, img_pred_bckgd)
            cntrs_bckgd = self.find_cntrs(img_pred_bckgd)
            if 1 in debug:
                print(f'nb of cntrs in the bckgd is {len(cntrs_bckgd)}')
            self.nb_false_pos_static = len(cntrs_bckgd)

            return img_bckgd, cntrs_bckgd

        except:
            print('Cannot calculate the nb of false positive detections')

            return None, None


    def find_nb_false_pos(self, f, img_bckgd, cntrs):
        '''
        '''
        nb_pos = 0
        shift_w_bckgd = self.find_shift_with_bckgd(f,img_bckgd)
        self.false_cntrs = self.shift_cntrs(self.false_cntrs, shift_w_bckgd)
        for c1 in self.false_cntrs:
            for c2 in cntrs:
                score = self.IoU_filter(c1, c2)
                if score > 0.8:
                    print(f'score is {score}')
                    nb_pos += 1

        return nb_pos

    def find_shift_with_bckgd(self, f,img_bckgd):
        '''
        '''
        img = cv2.imread(f)
        shift = self.mb.find_shift(img_bckgd, img)

        return np.array(shift)

    def shift_cntrs(self, cntrs, shift):
        '''
        '''
        print(f'shift the contours of the bckgrnd')
        new_cntrs = []
        for c in cntrs:
            new_c = []
            for v in c:
        #         print(v)
                new_c += [v+shift]
            new_cntrs += [new_c]

        return new_cntrs

    def pos_from_cntr(self, c):
        '''
        Extract the position from the contour
        find xy, xy, find_pos
        '''
        x, y, w, h = cv2.boundingRect(c)
        pos = (int(x + w/2), int(y + h/2))  # find position
        return pos

    def add_to_list_pos(self, i,cntrs):
        '''
        '''
        for c in cntrs:
            self.dic_pos[i].append(self.pos_from_cntr(c))

    def separate_groups(self):
        '''

        '''
        lpts = []
        # go through all the positions
        for k,v in self.dic_pos.items():
            for p in v:
                lpts += [[p[0],p[1]]]
        nlpts = np.array(lpts)

    def plot_all_pos(self, debug=[]):
        '''
        Plot the positions to extract the statistics..
        '''
        lpts = []
        if 0 in debug:
            print(f'len(self.dic_pos) is {len(self.dic_pos)}')
        for k,v in self.dic_pos.items():
            for p in v:
                if 0 in debug:
                    print(f'p[0], p[1] is {p[0], p[1]}')
                lpts += [[p[0], p[1]]]
        try:
            arr_pts = np.array(lpts)
            pk_addr = opj(self.folder_results, f'{self.well}_all_pts.pkl')
            pk.dump( arr_pts, open( pk_addr, "wb" ) )
            if 1 in debug:
                print(f'arr_pts[0:20] = {arr_pts[0:20]}')
            # Find optim gaussian mixture with correct clusters number..
            self.find_optim_gm(arr_pts)
            self.plot_optim_GM(arr_pts)
            self.plot_pts_distrib(arr_pts)
        except:
            print('Cannot plot the clusters..')

    def test_img_22(self, i, cntr, pkl=False):
        '''
        '''
        print('infos for image 22..')
        _, h, w, _ = self.img.shape
        mask = np.zeros((h, w, 3), np.uint8)
        print(f'len(cntr) is {len(cntr)}')
        print(f'cntr is {cntr}')
        cv2.drawContours(mask, [cntr], -1, (0, 255, 255), 1) #
        area_cnt = cv2.contourArea(cntr)
        if pkl:
            pk.dump( cntr, open( opj(self.folder_results,'cnt_clust_img22.pkl'), "wb" )  )

        print(f'area cnt is {area_cnt}')
        for c in self.lcntrs[i]:
            print(f'c is {c}')
            pt = self.pos_from_cntr(c)
            print(f'pos of pt is {pt}')
            area_pt = cv2.contourArea(c)
            print(f'area_pt is {area_pt}')
            # fill contour
            cv2.drawContours(mask, [c], -1, (255, 255, 255), -1)
            print(f'len(c) is {len(c)}')
        pos_cnt = self.pos_from_cntr(cntr)
        print(f'pos_cnt is {pos_cnt}')
        cv2.imwrite(opj(self.folder_results,
                        f'cnt with pos for 22.png'),
                    mask)

    def list_pts_inside(self, i, cntr, flip=False, test=False, debug=[]):
        '''
        Find the points inside the contour cnt..
        '''
        self.linside = []
        if flip:
            # flip vertically the cluster contour
            cntr = np.array([1,-1])*cntr + np.array([0,512])

        for c in self.lcntrs[i]:
            pt = self.pos_from_cntr(c)
            dist = cv2.pointPolygonTest(cntr, pt, False)
            if 0 in debug:
                print(f'*** dist is {dist}')
            if dist > 0:
                self.linside += [pt]
        if test:
            if i == 22:
                self.test_img_22(i,cntr)

    def no_div_before_lim(self, nbhours, filt_cntrs):
        '''
        No division before nbhours hours
        '''
        new_lpos = []
        for i,lpos in enumerate(filt_cntrs):
            if i < nbhours and len(lpos) > 2:
                lpos = []
            new_lpos += [lpos]
        filt_cntrs = new_lpos

    def find_maxi_area(self, lcntrs):
        '''
        '''
        max_area = 0
        max_cnt = None
        for i,cnt in enumerate(lcntrs):
            if cnt != []:
                area = cv2.contourArea(cnt)
                if area > max_area:
                    max_area = area
                    max_cnt = cnt
        print(f'max_area = {max_area}')
        return max_cnt

    def find_cntrs_in_cells_clusters(self, debug=[]):
        '''
        Find the cells inside the cluster area
        '''
        print('Dealing with counting in the cluster !!!')
        max_cnt = self.find_maxi_area(self.cntrs_clusters)
        if 0 in debug:
            print(f'max_cnt is {max_cnt}')
        for i in range(len(self.lcntrs)):
            self.list_pts_inside(i, max_cnt, flip=True)
            self.filtered_cntrs_stat += [self.linside]
        self.no_div_before_lim(20, self.filtered_cntrs_stat)
        if 1 in debug:
            print(f'self.filtered_cntrs_stat is {self.filtered_cntrs_stat}')

    def find_cntrs_in_cells_areas(self, debug=[]):
        '''
        Find the cells inside each cells area for each image
        '''
        # Go through all the predicted cells areas..
        # and find cells inside thoses areas..
        for i,cnt_area in enumerate(self.lcells_area):
            if 0 in debug:
                print(f'Dealing with cnt area {i}')
            # try:
            if cnt_area != []:
                # area where cells are detected with debris
                area = cv2.contourArea(cnt_area[0])
            else:
                area = 0
            if 0 in debug:
                print(f'area = {area}')
            if area > self.size_min_cloud:
                print(f'cnt big enough, area is {area}')
                self.list_pts_inside(i, cnt_area[0])
                prev_cnt_area = cnt_area[0]
            else:
                try:
                    # if area is bad use the last correct area for dscriminating..
                    self.list_pts_inside(i, prev_cnt_area)
                except:
                    self.linside = []

            # except:
            #     print('Probably a problem with contourArea')
            #     linside += self.lcntrs[i]
            self.filtered_cntrs += [self.linside]
            if 1 in debug:
                print(f'len(self.linside) = {len(self.linside)}')
            self.no_div_before_lim(20, self.filtered_cntrs)

    def filter_cntrs(self, debug=[0,1,3,4]):
        '''
        Keep only the contours cntrs inside the contour cnt..
        self.lcells_area : list of all registered cells areas (for image 0, 1 etc..)
        '''
        self.filtered_cntrs = []
        self.filtered_cntrs_stat = []
        if 2 in debug:
            print(f'len(self.lcells_area) is {len(self.lcells_area)}')
            print(f'self.lcells_area[:5] is {self.lcells_area[:5]}')
            print(f'len(self.lcntrs) is {len(self.lcntrs)}')
            print(f'self.lcntrs[:5] is {self.lcntrs[:5]}')
        self.find_cntrs_in_cells_areas()
        try:
            self.find_cntrs_in_cells_clusters()
        except:
            print('Cannot count the cells in the cells cluster.. ')
        # nb of cells counted without filtering
        self.lnbcells_orig = [len(cnts) for cnts in self.lcntrs]
        # nb of cells after filtering
        self.lnbcells = [len(cnts) for cnts in self.filtered_cntrs]
        # nb cells with stat method
        self.lnbcells_stat = [len(cnts) for cnts in self.filtered_cntrs_stat]
        if 4 in debug:
            print(f'### self.lnbcells = {self.lnbcells}')
            print(f'### self.lnbcells_stat = {self.lnbcells_stat}')

    def list_of_cells_area_contours(self, img_pred_area):
        '''
        '''
        # contours for stem cells area
        self.cntrs_area = self.find_cntrs(img_pred_area, thresh=1)
        # save contours of cells debris in list
        ll = [c  for c in self.cntrs_area if cv2.contourArea(c) > self.size_min_cloud] # else np.array([(0,0)])
        self.lcells_area += [ll]

    # def image_filtered_high(self, img_pred, size_cell_max=12):
    #     '''
    #     Remove shapes too large which may be some fusion..
    #     '''
    #     # contours for stem cells area
    #     kernel = np.ones((2,2),np.uint8)
    #     img_pred = cv2.erode(img_pred,kernel,iterations = 1)
    #     cntrs_detect = self.find_cntrs(img_pred, thresh=1)
    #     # save contours not too big
    #     filtered = [c  for c in cntrs_detect if cv2.contourArea(c) > size_cell_max]
    #     _, h, w, _ = self.img.shape
    #     mask = np.zeros((h, w), np.uint8)
    #     cv2.drawContours(mask, filtered, -1, (255, 255, 255), -1) # fill contour
    #
    #     return mask

    def make_composite_img(self, ind, list_img_pred, debug=[]):
        '''
        from images in list_img_pred, make the composite image cmp_img
        Make the fusion of the predictions contours
        '''
        _, h, w, _ = self.img.shape
        cmp_img = np.zeros((h, w), np.uint8) # composite


        for i, img_pred in enumerate(list_img_pred):
            # save the prediction used for fusion
            cv2.imwrite(opj(self.pred_folder,
                            f'img{ind}_pred_model{i}.png'),
                         img_pred)
            # remove when cells are fusionned by remove large shapes of cells..
            #filtered = self.image_filtered_high( np.squeeze(img_pred) )
            if 0 in debug:
                print(f'type(img_pred) is {type(img_pred)}')
                print(f'img_pred.shape is {img_pred.shape}')
                print(f'cmp_img.shape is {cmp_img.shape}')
            cmp_img[ np.squeeze(img_pred) > self.cmp_thr ] = 255
        # save the fusionned image..
        cv2.imwrite(opj(self.pred_folder,
                        f'img{ind}_models_fusion.png'),
                     cmp_img)

        return cmp_img

    def process_a_well(self,i,f, debug=[1]):
        '''
        '''
        self.dic_pos[i] = []
        print(f'current image is { f }')
        print(f'i is { i }')
        # if i%step_bckgd == 0:
        #     if nb_files-i > range_bckgd+1:
        #         img_bckgd, self.false_cntrs = self.find_false_pos_bckgd(i, range_bckgd)
        if 1 in debug:
            print(f'Will make predictions on well {self.well} for image {i}')
        self.img, list_img_pred, img_pred_area = self.make_pred(i, f)
        # contours from the multiple predictions
        cmp_img = self.make_composite_img(i, list_img_pred)
        # find contours on composite image
        cntrs = self.find_cntrs(cmp_img, filt_surface=True)
        self.lcntrs += [cntrs]
        self.list_of_cells_area_contours(img_pred_area)
        self.add_to_list_pos(i,cntrs)
        #self.nb_pos = self.find_nb_false_pos(f, img_bckgd, cntrs)
        # Save images of the contours : cells and cells area
        self.save_well_pics(i, self.img, cntrs, self.cntrs_area)


        ymdh = self.extract_date(f)
        self.ltimes += [ymdh]

    def count(self, time_range=None, debug=[0,2]):
        '''
        Count the number of cells in the images of given well (self.well)
        Go through all the consecutive images for this well.
        time_range : indices of the times to be processed..        '''
        if 0 in debug:
            print('In count !!!')
        if 1 in debug:
            print(f'self.addr_files is {self.addr_files}')
        self.lnbcells = []
        self.lnbcells_stat = []
        self.lcntrs = []
        self.lcells_area = []
        self.ltimes = []
        self.l_level = []
        self.curr_nb = 0
        self.nb_cells_max = 0
        nb_files = len(self.addr_files)
        step_bckgd = 20
        range_bckgd = 15
        self.size_min_cloud = 5e3
        self.dic_pos = {}

        if 1 in debug:
            print(f'On the brink to process the images for the well {self.well}')
        for i, f in enumerate(self.addr_files):
            if time_range:
                if i in time_range:
                    self.process_a_well(i,f)
            else:
                self.process_a_well(i,f)
        # plot all the positions for given well...
        self.plot_all_pos()
        # filter cells which are in cells area..
        self.filter_cntrs()
        self.lnbcells_levels = self.max_density_levels(np.array(self.lnbcells))[:len(self.lnbcells)]
        self.lnbcells_stat_levels = self.max_density_levels(np.array(self.lnbcells_stat))[:len(self.lnbcells)]
        # set to for comparing to annotations..
        self.lnbcells_stat_levels[self.lnbcells_stat_levels==1]=0

        print(f'len(self.lnbcells) = {len(self.lnbcells)}')
        # save the analyses
        self.save_result_in_dict('direct_ML')
        self.save_result_in_dict('stat')
        #self.save_nb_cells_max(i, cntrs)
        self.save_nb_cells_max_in_filtered()

    def full_list(self,ref,old_list, debug=[0]):
        '''
        '''
        newl = []
        for i,l in enumerate(ref):
            if i>0:
                prev = ref[i-1]
                curr = l
                prev_val = old_list[i-1]
                newl += [prev_val]*int((curr-prev)/self.delta_exp)
        return newl

    def score_with_level(self, vec_nb_cells, levels, debug=[0,1]):
        '''
        Score calculation between levels and annotations..
        '''
        vec_stat = np.array(levels)
        # # levels at 1 put at 0 ..
        # vec_stat[vec_stat==1]=0
        lenvec = len(vec_nb_cells)
        # adapt the length of vec_stat to length of manual annotations
        vec_stat = vec_stat[:lenvec]
        print(f'len(vec_stat) { len(vec_stat) }')
        print(f'len(vec_nb_cells) { len(vec_nb_cells) }')
        antiscore  = np.abs(vec_stat-vec_nb_cells).sum()/sum(vec_nb_cells)
        if 1 in debug:
            print(f'(vec_stat-vec_nb_cells).sum() = { (vec_stat-vec_nb_cells).sum() }')
            print(f'lenvec = {lenvec}')
            print(f'len(vec_stat) = {len(vec_stat)}')
            print(f'antiscore = {antiscore}')
        # score for comparison with manual annotation..
        curr_score = round((1-antiscore)*100,1)
        print(f'score for ML is {curr_score}')

        return curr_score

    def make_score(self, debug=[0,1,2]):
        '''
        Comparison with annotations
        '''
        # try:
        self.curr_score_ml = 0
        self.curr_score_stat = 0
        try:

            hours, nbcells = self.retrieve_times_nb_cells(self.well)
            print(f'hours, nbcells : {hours, nbcells}')
            try:
                ind_nan_min = np.argwhere(np.isnan(hours)).min()
                hours = hours[:ind_nan_min-1]
                nbcells = nbcells[:ind_nan_min-1]
            except:
                print('probably no NaN')
            lhours = sorted(list(set(hours)))
            lnbcells = list(set(nbcells))
            if 1 in debug:
                print(f'len(lhours) {len(lhours)}')
                print(f'len(lnbcells) {len(lnbcells)}')
            print('manual results in full list format.. ')
            vec_nb_cells = self.full_list(lhours,lnbcells)  # manual result
            if 1 in debug:
                print(f'In make_score, len(vec_nb_cells) is {len(vec_nb_cells)}')
            self.curr_score_stat = self.score_with_level(vec_nb_cells, self.lnbcells_stat_levels)
            self.curr_score_ml = self.score_with_level(vec_nb_cells, self.lnbcells_levels)
            # save the scores for the color in the interface
            self.save_scores()

        except:
            print('Cannot calculate the score..')

    def ins_pic(self, fig, img, pos_size, dic_txt=None, opacity=0.8):
        '''
        insert a picture
        '''
        nax = fig.add_axes(pos_size, anchor='NE', zorder=1)
        if dic_txt:
            txt = dic_txt['txt']
            pos = dic_txt['pos']
            img = self.insert_text(img, txt, pos=pos)
        nax.imshow(img, alpha=opacity)
        nax.axis('off')

    def calc_pos_size(self, ind, i, img_size=0.2):
        '''
        Make the list for position and size
        ind : index of the peak position
        i : index of the division
        '''
        nb_imgs = len(self.addr_files)
        #relx = ind/nb_imgs -0.1
        self.relx += 0.15                    # relative position on x axis
        rely = i*0.1                         # relative position on y axis
        pos_size = [self.relx, rely, img_size, img_size]

        return pos_size

    def prep_ins_img_orig(self, ind):
        '''
        '''
        addr_img = self.addr_files[ind]
        lw = 0.2                                     #  more width
        lh = 0.2                                     #  more height
        img = cv2.imread(addr_img)
        imgsh = img.shape
        c, d = int(imgsh[1]*(0.3-lw)), int(imgsh[1]*(0.5+lw))       # width
        a, b = int(imgsh[0]*(0.5-lh)), int(imgsh[0]*(0.7+lh))       # height
        print(f'a,b,c,d {a,b,c,d}')
        img = img[a:b, c:d]            # showing the center of the picture only

        return img

    def prep_ins_img_superp(self, ind):
        '''
        '''
        img_sup = cv2.imread(opj(self.pred_folder, f'img{ind}_superp.png'))

        return img_sup

    def ins_img_jumps(self, fig, insert_pic=False, debug=[]):
        '''
        Insert the images associated to the jumps
        '''
        if 0 in debug:
            print(f'self.list_jumps {self.list_jumps} ')
        self.relx = -0.15
        for i, ind in enumerate(self.list_jumps):

            img = self.prep_ins_img_orig(ind)
            img_superp = self.prep_ins_img_superp(ind)
            pos_size = self.calc_pos_size(ind, i)
            dic_txt = {'txt': self.lmdh[ind],                # insert day hour
                       'pos': (10, 50)}
            # BF picture
            self.ins_pic(fig, img, pos_size, dic_txt)
            pos_size[1] += 0.2
            # insert nb cells
            dic_txt = {'txt': 'nbcells ' + str(self.lnbcells[ind]),
                       'pos': (10, 50)}
            # picture with segmentation
            # BF + pred picture
            self.ins_pic(fig, img_superp, pos_size, dic_txt)

    def format_funcx(self, val, tick_nb, day_hour=False):
        '''
        Day and hours for x ticks
        '''
        try:
            if day_hour:
                return f'{self.lmdh[int(val)]}'
            else:
                return f'{int(val)}'
        except:
            return ''

    def format_funcy(self, val, tick_nb):
        '''
        Keeping the integer values only, for y axis
        '''
        if int(val) == val:
            return f'{int(val)}'
        else:
            return ''

    def make_axes(self, ax):
        '''
        Make the x and y axes for the analysis plot
        '''
        ## x axis
        plt.xticks(fontsize=self.fsize)
        ax.xaxis.set_major_formatter(plt.FuncFormatter(self.format_funcx))
        plt.xlabel('date', fontsize=self.fsize)
        ### y axis
        plt.yticks(fontsize=self.fsize)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(self.format_funcy))
        plt.ylabel('number of cells', fontsize=self.fsize)

    def make_time_axis(self,vec):
        '''
        '''
        time_axis = self.delta_exp*np.arange(len(vec))

        return time_axis

    def plot_levels(self, pic_at_jumps=False, debug=[0]):
        '''
        plot the results
        '''
        plt.figure()
        plt.rcParams["axes.edgecolor"] = "white"

        curr_well = f'well {self.well}, levels '
        # pic result dimensions
        pic_dims = 60
        fig, ax = plt.subplots(1, 1, figsize=(pic_dims, pic_dims))
        ###
        self.make_axes(ax)
        plt.grid()
        if 0 in debug:
            print(f'Length lnbcells_levels is { len(self.lnbcells_levels) }')
            print(f'Length lnbcells_stat_levels is { len(self.lnbcells_stat_levels) }')
            print(f'Length lnbcells_orig is { len(self.lnbcells_orig) }')

        # plot the nb of cells with time
        ta0 = self.make_time_axis(self.lnbcells_levels)
        ax.plot(ta0, np.array(self.lnbcells_levels) + 0.1, linewidth=10, label='nb cells after filtering')
        # plot the nb of cells with time with stat filtering method
        ta1 = self.make_time_axis(self.lnbcells_stat_levels)
        ax.plot(ta1, np.array(self.lnbcells_stat_levels) + 0.15, linewidth=10, linestyle='dashed', label='nb cells stat')

        # plot the nb of cells with time with no filtering
        ta2 = self.make_time_axis(self.lnbcells_orig)
        ax.plot(ta2, self.lnbcells_orig, linewidth=10, label='nb cells orig')
        ##
        ta3 = self.make_time_axis(self.lnbcells_stat)
        ax.plot(ta3, np.array(self.lnbcells_stat) + 0.15, linewidth=10, linestyle='dashed', label='nb cells after cluster stat filtering')
        # # guessing the real number of cells
        # ax.plot(self.l_level, linewidth=10, label='levels after filtering')
        try:
            # Manual annotations
            hours, nbcells = self.retrieve_times_nb_cells(self.well)
            print(f'hours, nbcells are {hours, nbcells}')
            ax.plot(hours, nbcells, linewidth=10, label='manual annotations')
        except:
            print(f'Cannot plot manual annotations for well num {self.well}')
        # insert picture for controlling the pred at jumps
        if pic_at_jumps:
            self.ins_img_jumps(fig)
        plt.title(f'{curr_well}\
                      score stat :{self.curr_score_stat}%,\
                      score ml :{self.curr_score_ml}%',\
                      fontsize=self.fsize)
        plt.legend(fontsize=self.fsize)
        plt.savefig(opj(self.folder_results, curr_well + '.png'), facecolor='w')

        pk.dump( np.array(self.lnbcells), open( opj(self.folder_results, f'nbcells {self.well}.pkl'), "wb" ) )

        ###

        plt.figure()
        nbiter = 2
        fig, ax = plt.subplots(1, 1, figsize=(pic_dims, pic_dims))
        self.make_axes(ax)
        ax.plot(np.array(self.lnbcells), linewidth=10, label='nb cells after filtering')
        ax.plot(np.array(self.lnbcells_stat), linewidth=10, label='nb cells after filtering with stat')
        ax.plot(np.array(self.lnbcells_levels) + 0.1, linewidth=10, label='levels ML')
        ax.plot(np.array(self.lnbcells_stat_levels) + 0.15, linewidth=10, label='levels Stat')
        # ax.plot(self.l_level, linewidth=10, label='levels after filtering')
        try:
            ax.plot(hours, nbcells, linewidth=10, label='manual annotations')
        except:
            print(f'Cannot plot manual annotations for well num {self.well}')
        curr_well = f'well {self.well}'
        plt.savefig(opj(self.folder_results, curr_well + '.png'))

    def analyse_one_well(self, well, time_range=None, name_well=True):
        '''
        Make an analysis of the number of cells in one well
        '''
        t0 = time()
        if name_well:
            print(f'current well is {well}')
        try:

            self.list_jumps = []                # list of the detected divisions
            self.lmdh = []                               # list day/hours
            self.pred_folder = opj(self.folder_results, f'pred_{well}')
            os.mkdir(self.pred_folder)               # prediction folder
            self.list_imgs(well=well)          # list of the images for one well
            self.count(time_range)               # count the nb of cells through the pictures
            self.make_score()
            self.plot_levels()                # show the result

        except:
            print(f'Cannot deal with well {well}')
        t1 = time()
        ttime = round((t1-t0)/60, 2)
        # time for one well..
        print(f'time for analysis of the well {well} is {ttime}')

    def plot_analysis_all_wells(self):
        '''
        Synoptic figure of the evolution of
         the number of cells in all the wells.
        '''
        print('Synthetic view of the results')
        pic_dims = 60
        fig, ax = plt.subplots(1, 1, figsize=(pic_dims, pic_dims))
        for k, v in self.dic_nbcells.items():
            print(v)
            ax.plot(v, label=k, linewidth=10)
            ax.legend(fontsize=self.fsize)
        plt.title('nb of cells in all the wells', fontsize=self.fsize)
        plt.xlabel('time', fontsize=self.fsize)
        ###
        plt.yticks(fontsize=self.fsize)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(self.format_funcy))
        plt.ylabel('#cells', fontsize=self.fsize)
        plt.grid()
        plt.savefig(opj(self.folder_results, 'all_wells.png'))

    def save_csv(self):
        '''
        Save all wells with the times
         and the number of cells in the "results" folder
        '''
        for kind_meth in ['stat', 'direct_ML']:
            self.csv(self.dic_tnbc[kind_meth], kind_meth)

    def reform_dict(self, nest_dict):
        '''
        Reorganize the dictionary for multiindex format
        '''
        ref_dict = {}
        for outk, indic in nest_dict.items():
            for ink, val in indic.items():
                ref_dict[(outk, ink)] = val

        return ref_dict

    def csv(self, data, kind_meth, debug=[]):
        '''
        Save the data as a csv file in results
        with the name of the experiment (eg:AD63) and the kind(eg:HSC)
        '''
        addr_csv = opj(self.folder_results, f'nbcells_{self.id_exp}-{self.kind_exp}-{kind_meth}.csv')
        # handle case where columns are not the same length
        df = pd.DataFrame({ k:pd.Series(val) for k, val in data.items() })
        df.to_csv(addr_csv, index=False, mode='w')
