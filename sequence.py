#!/usr/bin/python3

import matplotlib.pyplot as plt
import cv2
import numpy as np
from glob import glob
import os
import os.path as path
import sys, getopt
#import pyautogui

class Sequence:

    def __init__(self, seq_dir, verbose=False, zoom_fact=1.0, time_delay=1000, outdir=None):
        self.seq_dir = seq_dir
        self.parms = {}
        self.verbose = verbose
        self.zoom_fact = zoom_fact
        self.time_delay=time_delay
        self.outdir = outdir
        self.bgmask = None
        self.prev_mask = None
        
    def readSeqParms(self):
        filename = '%s/DET_PARMS.txt' %self.seq_dir
        try:
            f = open(filename, 'r')
        except IOError:
            print('Greska u citanju parametara:', filename)
            return None
        
        self.parms = {}
        self.parms['ROI_XS'] = 0
        self.parms['ROI_YS'] = 0
        self.parms['ROI_XE'] = -1
        self.parms['ROI_YE'] = -1
        
        self.parms['GIS_OFFSET_X'] = 0
        self.parms['GIS_OFFSET_Y'] = 0
        self.parms['GIS_OFFSET_A'] = 0
        
        for line in f:
            splitline=line.split()
            if len(splitline) > 0:
                if splitline[0] == 'START_INDEX:':
                    self.parms['START_INDEX'] = int(splitline[1])
                elif splitline[0] == 'END_INDEX:':
                    self.parms['END_INDEX'] = int(splitline[1])
                elif splitline[0] == 'MAX_INDEX:':
                    self.parms['MAX_INDEX'] = int(splitline[1])
                elif splitline[0] == 'IMG_WIDTH:':
                    self.parms['IMG_WIDTH'] = int(splitline[1])
                elif splitline[0] == 'IMG_HEIGHT:':
                    self.parms['IMG_HEIGHT'] = int(splitline[1])
                # ROI info
                elif splitline[0] == 'ROI_XS:':
                    self.parms['ROI_XS'] = int(splitline[1])
                elif splitline[0] == 'ROI_YS:':
                    self.parms['ROI_YS'] = int(splitline[1])
                elif splitline[0] == 'ROI_XE:':
                    self.parms['ROI_XE'] = int(splitline[1])
                elif splitline[0] == 'ROI_YE:':
                    self.parms['ROI_YE'] = int(splitline[1])
                # GIS PIXMAP info
                elif splitline[0] == 'GIS_OFFSET_X:':
                    self.parms['GIS_OFFSET_X'] = int(splitline[1])
                elif splitline[0] == 'GIS_OFFSET_Y:':
                    self.parms['GIS_OFFSET_Y'] = int(splitline[1])
                elif splitline[0] == 'GIS_OFFSET_A:':
                    self.parms['GIS_OFFSET_A'] = int(splitline[1])

        # Ako nema ROI, stavljam na veci dio slike
        if self.parms['ROI_XE'] == -1:
            self.parms['ROI_XS'] = 30
            self.parms['ROI_XE'] = self.parms['IMG_WIDTH'] - 30
        if self.parms['ROI_YE'] == -1:
            self.parms['ROI_YS'] = 30
            self.parms['ROI_YE'] = self.parms['IMG_HEIGHT'] - 30

        f.close()
        return self.parms

    def checkParms(self):
        if len(self.parms) != 5:             # Parametri nisu procitani
            if self.readSeqParms() == None:  # Greska u citanju
                print('Neispravna datoteka s parametrima')
                return False

        return True

    def printSeqInfo(self):
        if not self.checkParms():   # Provjeravamo ispravnost parametara
            return None
            
        print('IMG_WIDTH:  \t %5d' %self.parms['IMG_WIDTH']) 
        print('IMG_HEIGHT: \t %5d' %self.parms['IMG_HEIGHT']) 
        print('MAX_INDEX:  \t %5d' %self.parms['MAX_INDEX']) 
        print('START_INDEX:\t %5d' %self.parms['START_INDEX']) 
        print('END_INDEX:  \t %5d' %self.parms['END_INDEX']) 
    
    def printSeqROI(self):
        if not self.checkParms():   # Provjeravamo ispravnost parametara
            return None
        print('ROI:  \t\t (%5d, %5d) <---> (%5d, %5d)' \
              %(self.parms['ROI_XS'], self.parms['ROI_YS'],
                self.parms['ROI_XE'], self.parms['ROI_YE']))
    
    def createSequenceWindow(self, win_name='Sekvenca'):
        # pass
        self.win_name = win_name
        if self.zoom_fact == 0:
            cv2.namedWindow(self.win_name, cv2.WINDOW_GUI_NORMAL | \
                            cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
            sw, sh = pyautogui.size()
            cv2.resizeWindow(self.win_name, sw, sh)
        else:
            cv2.namedWindow(self.win_name, cv2.WINDOW_GUI_NORMAL | \
                            cv2.WINDOW_NORMAL | cv2.WINDOW_AUTOSIZE)

            
    def readBGMask(self):
        maskfile = '%s/bgmask.jpg' %self.seq_dir
        if not os.path.isfile(maskfile):
            print("Ne postoji BGMASK maska")
            return
        
        wmask = cv2.imread(maskfile, cv2.IMREAD_GRAYSCALE)
        if wmask is not None:
            if wmask.shape[0] == self.parms['IMG_HEIGHT'] and wmask.shape[1] == self.parms['IMG_WIDTH']:
                _, self.bgmask = cv2.threshold(wmask, 127, 1,
                                               cv2.THRESH_BINARY)
                if self.zoom_fact != 1.0 and self.zoom_fact > 0:
                    self.bgmask = cv2.resize(self.bgmask, None,
                                             fx = self.zoom_fact,
                                             fy = self.zoom_fact,
                                             interpolation = cv2.INTER_NEAREST)
            elif self.verbose:
                print('Pogrešne dimenzije BGMASK maske')
        elif self.verbose:
            print("Greska u ucitavanju BGMASK maske")

    # Samo provjerava postoji li datoteka
    def checkPixinfo(self, pixinfo_file='pixinfo.yml'):
        infofile = '%s/%s' %(self.seq_dir, pixinfo_file)
        if os.path.isfile(infofile):
            return True
        else:
            return False
        
    def readPixinfo(self, pixinfo_file='pixinfo.yml'):
        infofile = '%s/%s' %(self.seq_dir, pixinfo_file)
        if not os.path.isfile(infofile):
            return None

        try:
            fs = cv2.FileStorage(infofile, cv2.FILE_STORAGE_READ)
        except:
            return None
        pixinfo = fs.getNode('pixinfo').mat()
        fs.release()
        return pixinfo

        
    def readFrame(self, cur_index, frindex):
        framefile = '%s/image_%05d-frame-%02d.jpg' \
                    %(self.seq_dir, cur_index, frindex)
        if not os.path.isfile(framefile):
            return None
        frame = cv2.imread(framefile)
        if frame is None:
            return None
        
        if self.zoom_fact != 1.0 and self.zoom_fact > 0:
            frame = cv2.resize(frame, None,
                               fx = self.zoom_fact, fy = self.zoom_fact,
                               interpolation = cv2.INTER_CUBIC)
        
        return frame

    def readMask(self, cur_index, frindex):
        maskfile = '%s/mask_%05d-frame-%02d.png' \
                   %(self.seq_dir, cur_index, frindex)
        mask = cv2.imread(maskfile, cv2.IMREAD_GRAYSCALE)
        
        if mask is None:
            if self.prev_mask is None:
                rows, cols = self.frame.shape[:2]
                mask = np.zeros((rows, cols), dtype=np.uint8)
            else:
                mask = self.prev_mask.copy()
        elif self.zoom_fact != 1.0 and self.zoom_fact > 0:
            mask = cv2.resize(mask, None,
                              fx = self.zoom_fact, fy = self.zoom_fact,
                              interpolation = cv2.INTER_NEAREST)
            
        if self.frame.shape[:2] != mask.shape[:2]:
            rows, cols = self.frame.shape[:2]
            mask = np.zeros((rows, cols), dtype=np.uint8)
            print('KRIVE DIMENZIJE: %s' %maskfile)
            
        return mask
        
    def writeFrame(self, cur_index, frindex):
        if self.outdir is None:
            return
        outframe = '%s/image_%05d-frame-%02d.jpg' \
                    %(self.outdir, cur_index, frindex)
        cv2.imwrite(outframe, self.wframe)

    
    def showMaskOnFrame(self):
        wmask = self.mask.copy()
        #_, contours, _ = cv2.findContours(wmask, cv2.RETR_LIST,
        #                                  cv2.CHAIN_APPROX_SIMPLE)
        contours, _ = cv2.findContours(wmask, cv2.RETR_LIST,
                                          cv2.CHAIN_APPROX_SIMPLE)
        self.wframe = self.frame.copy()
        cv2.drawContours(self.wframe, contours, -1, (255,0,0))
        
    def showFrame(self):
        cv2.imshow(self.win_name, self.wframe)
        #print(cv2.getWindowImageRect(self.win_name)) #DEV TEST
        #cv2.imshow('w2', self.mask)
        
    def createEmptyMasks(self):
        if not self.checkParms():   # Provjeravamo ispravnost parametara
            return None
        
        cur_index = self.parms['START_INDEX']
        end_index = self.parms['END_INDEX']
        max_index = self.parms['MAX_INDEX']
        
        seq_running = True
        
        while seq_running:
            frindex = 0
            while True:
                self.frame = self.readFrame(cur_index, frindex)
                if self.frame is None:
                    break
                
                mask = self.readMask(cur_index, frindex)
                if mask is None:
                    seq_running = False
                    break
                maskfile = '%s/mask_%05d-frame-%02d.png' \
                           %(self.seq_dir, cur_index, frindex)
                cv2.imwrite(maskfile, mask)
                
                frindex += 1
                    
            if self.verbose:
                print('---------')

            cur_index += 1
            if cur_index > max_index:
                cur_index = 1
            if cur_index > end_index:
                seq_running = False

       
        
    def playSequence(self):
        if not self.checkParms():   # Provjeravamo ispravnost parametara
            return None

        cur_index = self.parms['START_INDEX']
        end_index = self.parms['END_INDEX']
        max_index = self.parms['MAX_INDEX']
        self.createSequenceWindow()

        seq_running = True
        while seq_running:
            frindex = 0
            while True:
                self.frame = self.readFrame(cur_index, frindex)
                if self.frame is None:
                    break

                self.mask = self.readMask(cur_index, frindex)
                if self.mask is None:
                    seq_running = False
                    break

               
                if self.verbose:
                    print('frame: %5d, %2d' %(cur_index, frindex))

                self.showMaskOnFrame()    
                self.showFrame()
                if cv2.waitKey(self.time_delay) == 27:
                    seq_running = False
                    break
                if self.outdir is not None:
                    self.writeFrame(cur_index, frindex)
                    
                frindex += 1
                del self.frame
                del self.mask
                    
            if self.verbose:
                print('---------')

            cur_index += 1
            if cur_index > max_index:
                cur_index = 1
            if cur_index > end_index:
                seq_running = False

        cv2.destroyAllWindows()

# Direktno startanje iz komandne linije 
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Koristenje: %s -i <input_dir> [-v -z zoom]' %sys.argv[0])
        exit()

    verbose = False
    zoom_fact = 1.0
    seq_dir = None
    outdir = None
    time_delay=1000
    try:
        opts, args = getopt.getopt(sys.argv[1:], "hvi:z:o:p:",
                                   ["verbose", "input=","zoom=","output=","pause="])
    except getopt.GetoptError:
        print('Koristenje: %s -i <input_dir> [-h -v -z zoom -p pause]' %sys.argv[0])
        sys.exit(2)
        
    for opt, arg in opts:
        if opt == '-h':
            print('Koristenje: %s -i <input_dir> [-h -v -z zoom -p pause -o <outdir>]' %sys.argv[0])
            sys.exit()
        elif opt in ("-i", "--input"):
            seq_dir = arg
        elif opt in ("-o", "--output"):
            outdir = arg
        elif opt in ("-v", "--verbose"):
            verbose = True
        elif opt in ("-z", "--zoom"):
            zoom_fact = float(arg)
        elif opt in ("-p", "--pause"):
            time_delay = int(arg)

    if seq_dir is None:
        print('Koristenje: %s -i <input_dir> [-h -v -z zoom -o <outdir>]' %sys.argv[0])
        sys.exit()
            
    seq = Sequence(seq_dir, verbose=verbose, zoom_fact=zoom_fact, outdir=outdir, time_delay=time_delay)
    
    seq.printSeqInfo()
    seq.printSeqROI()
    seq.playSequence()
