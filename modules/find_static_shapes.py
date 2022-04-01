from skimage.metrics import structural_similarity
import cv2
import numpy as np

class FIND_STATIC():
    '''
    '''
    def __init__(self):
        '''
        '''
        self.area_size_min = 50
        self.area_size_max = 3e3

    def comp_imgs(self, img1, img2, i, show=False):
        '''
        '''
        self.prepare_imgs(img1, img2)
        self.make_comparison()
        cntrs = self.find_contours()
        if show:
            self.show_result()
        self.save_result(i)
        return cntrs

    def prepare_imgs(self, img1, img2, debug=[0]):
        '''
        '''
        if 0 in debug:
            print('prepare_imgs')
        self.before = cv2.imread(img1)
        self.after = cv2.imread(img2)

        # Convert images to grayscale
        self.before_gray = cv2.cvtColor(self.before, cv2.COLOR_BGR2GRAY)
        self.after_gray = cv2.cvtColor(self.after, cv2.COLOR_BGR2GRAY)

    def make_comparison(self, debug=[0]):
        '''
        '''
        if 0 in debug:
            print('make_comparison')
        # Compute SSIM between two images
        (self.score, self.diff) = structural_similarity(self.before_gray, self.after_gray, full=True)
        print("Image similarity", self.score)

        # The diff image contains the actual image differences between the two images
        # and is represented as a floating point data type in the range [0,1]
        # so we must convert the array to 8-bit unsigned integers in the range
        # [0,255] before we can use it with OpenCV
        self.diff = (self.diff * 255).astype("uint8")

    def find_contours(self, debug=[0]):
        '''
        Find the contours for prohibited area..
        '''
        if 0 in debug:
            print('find_contours')
        # Threshold the difference image, followed by finding contours to
        # obtain the regions of the two input images that differ
        thresh = cv2.threshold(self.diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        cntrs = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cntrs = cntrs[0] if len(cntrs) == 2 else cntrs[1]

        self.mask = np.zeros(self.before.shape, dtype='uint8')
        self.filled_after = self.after.copy()

        for c in cntrs:
            area = cv2.contourArea(c)
            if self.area_size_max > area > self.area_size_min:
                x,y,w,h = cv2.boundingRect(c)
                cv2.rectangle(self.before, (x, y), (x + w, y + h), (36,255,12), 2)
                cv2.rectangle(self.after, (x, y), (x + w, y + h), (36,255,12), 2)
                cv2.drawContours(self.mask, [c], 0, (0,255,0), -1)
                cv2.drawContours(self.filled_after, [c], 0, (0,255,0), -1)

        return cntrs

    def show_result(self, debug=[0]):
        '''
        '''
        if 0 in debug:
            print('show results')
        cv2.imshow('before', self.before)
        cv2.imshow('after', self.after)
        #cv2.imshow('diff', self.diff)
        #cv2.imshow('mask', self.mask)
        cv2.imshow('filled after',self.filled_after)
        cv2.waitKey(0)

    def save_result(self, i, debug=[0]):
        '''
        '''
        cv2.imwrite(f'static_patterns_{i}.png', self.filled_after)
