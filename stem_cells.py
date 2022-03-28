'''
Program for segmenting and counting the stem cells
'''

import glob
import re
import os
import numpy as np

import yaml
import shutil as sh
from pathlib import Path
import pandas as pd
from matplotlib import pyplot as plt
from tensorflow.keras import models
from modules.find_static_shapes import FIND_STATIC as FS

import cv2
op = os.path
opb, opd, opj = op.basename, op.dirname, op.join


class STEM_CELLS(FS):
    '''
    Detect and count the stem cells
    The program uses Tensorflow
    '''

    def __init__(self, addr_folder, model='stem_ep30', min_area=2):
        '''
        addr_folder: address of the images
        model : model shortcut used for counting the cells
        '''
        FS.__init__(self)

        self.curr_model = model
        self.addr_folder = addr_folder                 # path for data
        self.id_exp = opb(opd(self.addr_folder))
        self.kind_exp = opb(self.addr_folder)
        self.folder_results = f'results_mod_{self.curr_model}'\
                              f'_data_{self.id_exp}-{self.kind_exp}'\
                              f'_{self.date()}'
        with open('models.yaml') as f:
            dic_models = yaml.load(f, Loader=yaml.FullLoader)
            curr_mod = dic_models[model]       # full name of the current model
        addr_model = Path('models') / curr_mod

        print(addr_model)
        self.mod_stem = models.load_model(addr_model)        # load the model
        self.prepare_result_folder()        # create the folder for the results
        self.min_area = min_area            # minimal area for cell
        self.max_area = 20                  # maximal area for cell
        self.lev_thr = 100                  # threshold level for segmentation
        self.dic_nbcells = {}
        self.dic_tnbc = {'well':[],
                         'time':[],
                         'nbcells':[]}      # dictionary of time and nb cells
        self.list_tnbc = []                 # list of time and nb cells

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
        with open(f'{self.folder_results}/proc_infos.yaml', 'w') as f_w:
            # save model name in proc_infos.yaml
            dic_infos = {'dataset': self.last_two_levels(self.addr_folder),
                         'model': self.curr_model}
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

    def list_imgs(self, well=None, debug=[0]):
        '''
        List of the images for a given well
        '''
        self.well = well
        glob_string = f'{self.addr_folder}/*_{well}_*.tif'
        if 0 in debug:
            print(f'glob_string is {glob_string}')
        self.addr_files = glob.glob(glob_string)
        self.addr_files.sort(key=lambda x: self.mdh_to_nb(x))
        self.lmdh.sort(key=lambda x: self.mdh_to_nb(x, make_lmdh=False))

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

    def make_mask_from_contour(self,cnt, dilate=False, iter_dilate=1):
        '''
        '''

        h, w, nchan = self.img.shape
        mask = np.zeros((h, w), np.uint8)
        cv2.drawContours(mask, [cnt], -1, (255, 255, 255), -1) # fill contour

        return mask

    def IoU_filter(self,c1, c2):
        '''
        Compare Intersection over Union..
        '''
        mask0 = self.make_mask_from_contour(c1)                      # previous contour
        mask1 = self.make_mask_from_contour(c2)                           # new contour..
        inter = np.logical_and(mask0, mask1)
        union = np.logical_or(mask0, mask1)
        iou_score = np.sum(inter) / np.sum(union)
        print(f"### IoU score for  is {iou_score}")
        return iou_score

    def find_cntrs(self, img, debug=[]):
        '''
        Find the contours in the thresholded prediction
        '''
        try:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        except:
            gray = img
        ret, thr = cv2.threshold(gray, self.lev_thr, 255, 0)      # Threshold
        thr = thr.astype('uint8')
        cntrs, _ = cv2.findContours(thr, cv2.RETR_TREE,
                                    cv2.CHAIN_APPROX_SIMPLE)[-2:]
        if len(cntrs) > 7:
            for c in cntrs:
                if 1 in debug:
                    print(f'cv2.contourArea(c) is {cv2.contourArea(c)}')
        filt_cntrs = [c for c in cntrs if self.max_area
                      > cv2.contourArea(c) > self.min_area]

        filt_not_static = []
        for c1 in filt_cntrs:
            score_max = 0
            for c2 in self.prohibited_cntrs:
                score = self.IoU_filter(c1, c2)
                if score > score_max:
                    score_max = score
            if score_max < 0.1:
                filt_not_static += [c1]
        filt_cntrs = filt_not_static

        return filt_cntrs

    def make_levels(self, debug=[]):
        '''
        Number of cells supposed to be an even number
        '''
        if self.nbcntrs >= 2*self.curr_nb:
            if self.nbcntrs == 1:
                self.curr_nb = 1
                self.list_jumps += [self.curr_ind]
            elif self.nbcntrs > 1:
                self.curr_nb *= 2
                self.list_jumps += [self.curr_ind]
        print(f'self.curr_ind {self.curr_ind}')
        if 0 in debug:
            print(f'self.list_jumps {self.list_jumps}')
        if 1 in debug:
            print(f'self.curr_nb {self.curr_nb}')
        self.l_level += [self.curr_nb]

    def draw_pred_contours(self, img, cntrs, debug=[]):
        '''
        '''
        if 0 in debug:
            print(f'img.shape { img.shape }')
        _, h, w, _ = img.shape
        mask = np.zeros((h, w), np.uint8)
        # fill contour
        cv2.drawContours(mask, cntrs, -1, (255, 255, 255), -1)
        return mask

    def save_pred(self, i, img, cntrs):
        '''
        '''
        img_cntrs = self.draw_pred_contours(img, cntrs)
        cv2.imwrite(opj(self.pred_folder, f'img{i}_pred.png'), img_cntrs)

    def save_BF(self, i, cntrs):
        '''
        '''
        cv2.imwrite(opj(self.pred_folder, f'img{i}.png'), self.img_orig)

    def save_BF_superp(self, i, cntrs):
        '''
        '''
        try:
            cv2.drawContours(self.img_orig, cntrs, -1, (0, 0, 255), -1)
        except:
            print('Probably no contours')
        cv2.imwrite(opj(self.pred_folder,
                    f'img{i}_superp.png'), self.img_orig)

    def save_well_pics(self, i, img, cntrs):
        '''
        '''
        # save predictions contours..
        self.save_pred(i, img, cntrs)
        # save BF
        self.save_BF(i, cntrs)
        # save BF + pred
        self.save_BF_superp(i, cntrs)

    def make_pred(self, i, f, debug=[]):
        '''
        '''
        self.curr_ind = i
        # prepare images
        img = self.prep_img(f)
        # prediction with model stem_cells
        pred = self.mod_stem.predict(img)
        if 0 in debug:
            print(f'type(pred[0]) is {type(pred[0])}')
            print(f'pred[0].shape) is {pred[0].shape}')
        pred_img = pred[0]*255
        pred_img = pred_img.astype('uint8')
        try:
            img_pred = cv2.cvtColor(pred_img, cv2.COLOR_BGR2RGB)
        except:
            img_pred = pred_img

        return img, img_pred

    def save_nb_cells_max(self, i, cntrs):
        '''
        Save the number of cells detected..
        '''
        if self.nb_cells_max < len(cntrs):
            self.nb_cells_max = len(cntrs)
        if i == len(self.addr_files)-1:
            file_nb_cntrs = opj(self.pred_folder, f'nb_cells_max.yaml')
            with open(file_nb_cntrs, 'w') as f_w:
                yaml.dump(self.nb_cells_max, f_w)

    def extract_date(self, f):
        '''
        '''
        nimg = opb(f)
        ymdh = re.findall(r'\d+y\d+m\d+d_\d+h', nimg)[0]

        return ymdh

    def save_result_in_dict(self):
        '''
        Save the results in the dict which will be converted to csv
        '''
        # self.dic_tnbc['well_' + self.well] = [self.well]*len(self.ltimes)
        # self.dic_tnbc['time_' + self.well] = self.ltimes
        # self.dic_tnbc['nbcells_' + self.well] = self.lnbcells
        ##
        self.dic_tnbc['well'] += [self.well]*len(self.ltimes)
        self.dic_tnbc['time'] += self.ltimes
        self.dic_tnbc['nbcells'] += self.lnbcells

    def count(self, debug=[]):
        '''
        Count the number of cells in the images
        '''
        if 0 in debug:
            print('In count !!!')
            print(f'self.addr_files is {self.addr_files}')
        self.lnbcells = []
        self.ltimes = []
        self.l_level = []
        self.curr_nb = 0
        self.nb_cells_max = 0
        for i, f in enumerate(self.addr_files):
            print(f'current image is { f }')
            img, img_pred = self.make_pred(i, f)
            if i > 1:
                self.prohibited_cntrs = self.comp_imgs(self.addr_files[i-1],
                                                self.addr_files[i])
            cntrs = self.find_cntrs(img_pred)       # contours from predictions



            self.save_well_pics(i, img, cntrs)
            self.save_nb_cells_max(i, cntrs)
            self.nbcntrs = len(cntrs)
            self.lnbcells += [self.nbcntrs]
            ymdh = self.extract_date(f)
            self.ltimes += [ymdh]
            self.make_levels()
        print(f'len(self.lnbcells) = {len(self.lnbcells)}')
        # save the analyses
        self.save_result_in_dict()

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

    def format_funcx(self, val, tick_nb):
        '''
        Day and hours for x ticks
        '''
        try:
            return f'{self.lmdh[int(val)]}'
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

    def plot(self, pic_at_jumps=False):
        '''
        plot the results
        '''
        # font size
        self.fsize = 60
        curr_well = f'well {self.well}, threshold {self.lev_thr} '
        # pic result dimensions
        pic_dims = 60
        fig, ax = plt.subplots(1, 1, figsize=(pic_dims, pic_dims))
        ###
        plt.title(curr_well, fontsize=self.fsize)
        self.make_axes(ax)
        plt.grid()
        # plot the nb of cells with time
        ax.plot(self.lnbcells, linewidth=10)
        # guessing the real number of cells
        ax.plot(self.l_level, linewidth=10)
        # insert picture for controlling the pred at jumps
        if pic_at_jumps:
            self.ins_img_jumps(fig)

        plt.savefig(opj(self.folder_results, curr_well + '.png'))

    def analyse_one_well(self, well, name_well=True):
        '''
        Make an analysis of the number of cells in one well
        '''
        if name_well:
            print(f'current well is {well}')
        self.list_jumps = []                # list of the detected divisions
        self.lmdh = []                               # list day/hours
        self.pred_folder = opj(self.folder_results, f'pred_{well}')
        os.mkdir(self.pred_folder)               # prediction folder
        self.list_imgs(well=well)          # list of the images for one well
        self.count()               # count the nb of cells through the pictures
        self.plot()                # show the result

    def plot_analysis_all_wells(self):
        '''
        Synoptic figure of the evolution of
         the number of cells in all the wells.
        '''
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
        #self.csv(self.reform_dict(self.dic_tnbc))
        self.csv(self.dic_tnbc)
        #self.csv(self.list_tnbc)

    def reform_dict(self, nest_dict):
        '''
        Reorganize the dictionary for multiindex format
        '''
        ref_dict = {}
        for outk, indic in nest_dict.items():
            for ink, val in indic.items():
                ref_dict[(outk, ink)] = val

        return ref_dict

    def csv(self, data, debug=[]):
        '''
        Save the data as a csv file in results
        '''
        addr_csv = opj(self.folder_results, 'nbcells.csv')
        # handle case where columns are not the same length
        df = pd.DataFrame({ k:pd.Series(val) for k, val in data.items() })
        df.to_csv(addr_csv, index=False, mode='w')
