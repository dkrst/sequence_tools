import cv2
import numpy as np

class ForegroundExtractor:
    def __init__(self, alpha=0.9, t=0.05):
        self.a = alpha
        self.t = t
        self.T = None

    def apply(self, frame, step=False):
        wframe = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)/255.0
        shape = wframe.shape
        
        if self.T is None:  # First frame in sequence
            self.T = np.full(shape, self.t, dtype=np.float32)
            self.BG = wframe.copy()
            return np.zeros(shape, dtype = np.uint8)

        if step:
            #self.BG = .5*self.BG + .5*wframe
            self.BG = wframe.copy()
            FG = np.zeros(shape, dtype = np.uint8)
        else:
            self.D = np.abs(wframe - self.BG)
            FG = (self.D > self.T).astype(np.uint8)
            kernel = np.ones((5,5),np.uint8)
            FG = cv2.morphologyEx(FG, cv2.MORPH_OPEN, kernel)
            contFG = (1 - self.a) * self.BG * FG # FG contribution
            contBG = (1 - self.a) * wframe * (1 - FG) ## BG contrib.
            self.BG = self.a * self.BG + contFG + contBG
            # ILI (isto):
            '''
            self.BG = self.FG*self.BG + (1-self.FG)*(self.a*self.BG +
                                                     (1-self.a)*wframe)
            '''

            # Prag T raste tamo gdje je velika razlika, pada gdje je mala
            self.T = self.a * self.T + (1 - self.a) * self.D
            self.T = np.maximum(self.T, 0.7*self.t)
            
        return FG*255

    def getThresholdImg(self):
        return cv2.normalize(self.T, None, 0.0, 1.0, cv2.NORM_MINMAX)

    def getDiffImg(self):
        return cv2.normalize(self.D, None, 0.0, 1.0, cv2.NORM_MINMAX)
