import cv2
import numpy as np
#import string
#import random
#import shutil

MIN_SMOKE_PIX = 64

class SampleProcessor():
    def __init__(self, samples_dir):
        self.samples_dir = samples_dir
        self.drawing = False
        self.erasing = False
        
    def readSample(self, sample_name):
        # still
        filename = f'{self.samples_dir}/still/{sample_name}.npz'
        self.img = np.load(filename)['im']
        # temporal
        filename = f'{self.samples_dir}/temporal/{sample_name}.npz'
        self.tem = np.load(filename)['im']
        # dist
        filename = f'{self.samples_dir}/dist/{sample_name}.npz'
        self.dist = cv2.split(np.load(filename)['im'])[3] # cetvrti kanal
        # labels
        h, w = self.img.shape[:2]
        self.mask = np.zeros((h, w), dtype=np.uint8)
        filename = f'{self.samples_dir}/labels/{sample_name}.txt'
        with open(filename, 'r') as f:
            lines = f.readlines()
            for line in lines:
                cnt = line.split()
                xc = int(float(cnt[1]) * w)
                yc = int(float(cnt[2]) * h)
                sw = int(float(cnt[3]) * w)
                sh = int(float(cnt[4]) * h)
                xs = int(xc - sw/2)
                xe = xs + sw
                ys = int(yc - sh/2)
                ye = ys + sh
                cv2.rectangle(self.mask, (xs,ys), (xe,ye), 255, cv2.FILLED)
                cv2.rectangle(self.mask, (xs,ys), (xe,ye), 255, cv2.FILLED)

    def showSample(self):
        cv2.imshow('STILL', self.wimg)
        cv2.imshow('TEMPORAL', self.wtem)
        cv2.imshow('DIST', self.dist)
        
    def showMaskOnSample(self):
        wmask = self.mask.copy()
        contours, _ = cv2.findContours(wmask, cv2.RETR_LIST,
                                       cv2.CHAIN_APPROX_SIMPLE)
        # still
        self.wimg = self.img.copy()
        cv2.drawContours(self.wimg, contours, -1, (255,0,0))
        # temporal
        self.wtem = self.tem.copy()
        cv2.drawContours(self.wtem, contours, -1, (255,0,0))
        self.showSample()
        
    def showDistOnSample(self):
        cnt = cv2.Laplacian(self.dist, 5)
        
        # still
        self.wimg = self.img.copy()
        # temporal
        self.wtem = self.tem.copy()

        h, w = self.img.shape[:2]
        for i in range(h):
            for j in range(w):
                if cnt[i, j] > 0:
                    self.wimg[i,j,1] = 255
                    self.wtem[i,j,1] = 255
        self.showSample()
        
    def processSample(self, sample_name):
        self.readSample(sample_name)
        cv2.namedWindow('STILL', cv2.WINDOW_GUI_NORMAL | \
                        cv2.WINDOW_NORMAL | cv2.WINDOW_AUTOSIZE)

        self.showMaskOnSample()
        key = cv2.waitKey(0)
        cv2.destroyAllWindows()
        self.showDistOnSample()
        key = cv2.waitKey(0)
        cv2.destroyAllWindows()
        
