#!/home/dkrst/.venv/yoloset/bin/python3

#
# Generiranje seta slika sa dimom za YOLO algoritam
#
# Sekvence se obradjuju pojedinacno, svaka po vise puta
# sa augmentacijom
# Obradjuju se izvorne slike, prije racunanja dodatnih kanala
# (temporal, foreground, ...)
# geometrijske transformacije su izdvojene da bi se mogle
# primjeniti na maske i na pixel mape
#
# Generiram i set slika bez dima za validation
#
#

import matplotlib.pyplot as plt
import cv2
import numpy as np
from glob import glob
import os
import os.path as path
import sys
import json
import random
import sys, getopt
import shutil

sys.path.append('.')
from sequence import Sequence
from foreground_extractor import ForegroundExtractor
from albumentationsProcessor import AlbumentationsProcessor as alb


# Pixinfo data
CH_PIXSIZE = 0
CH_PIXDIST = 1
GIS_HORIZONT = -1

DIST_MAX = 25000.0
DIST_HORIZONT = 28000.0

MIN_SMOKE_PIX = 64
#ROI_XSIZE = 1920 # Default za YOLOv5
#ROI_YSIZE = 1024
ROI_XSIZE = 640
ROI_YSIZE = 640

#Najveca +/- vrijednst koju moze dati sobel 3x3
SOBEL_MAX = 1020
MAG_MAX = np.sqrt(2 * SOBEL_MAX*SOBEL_MAX) # Maksimalna magnituda

# Koliko frameova preskace kad najde dim
# (za >15 sigurno ide na sljedeci prolaz)
SKIP_FRAMES = 3

# Preskacemo do ovoliko linija slike s vrha (uglavnom nebo)
# pri izradi NOSMOKE seta
SKIP_TOP = 120
# Preskacemo do ovoliko linija slike s dna (preblizu)
# pri izradi NOSMOKE seta
SKIP_BOTTOM = 30
# Preskacemo rubove slike do max stupaca
SKIP_HORIZONTAL = 30

gen_samples = {
    'still': False,      # Still (original) image   3 CH(RGB)
    'temporal': False,   # Temporal image           3 CH
    'join_st': False,   # Still:Temporal           5 CH(B ne treba duplo)
    'dist': False,       # Still:Dist               4 CH
    'join_std': False,  # Still:Temporal:Dist      7 CH
    'fgr': False,       # Still:Fgr                5 CH
    'grad': False,      # Still:Grad               5 CH
    'gpolar': False,    # Still:GradPolar          5 CH
    'join_all': True,  # All                     12 CH
}

class AugmentedYoloSet(Sequence):
    def __init__(self, seq_dir, outdir):
        super(AugmentedYoloSet, self).__init__(seq_dir=seq_dir)
        # Ucitavam odmah parametre sekvence iz datoteke
 
        self.outdir = outdir
        self.num_samples =  {'smoke': 0, 'nosmoke':0}
        self.sequence_has_smoke = False
        self.fgextractor = ForegroundExtractor()

        self.transform = None
        self.aug_dist = None
        
    def setAugmentation(self, tprob = None):
        self.transform = alb()
        if tprob is None:
            self.transform.createSequenceTransform()
        else:
            self.transform.createSequenceTransform(
                scale = tprob['scale'],
                random_scale = tprob['random_scale'],
                horizontal_flip = tprob['horizontal_flip'],
                rotate = tprob['rotate'],
                perspective = tprob['perspective'],
                optical_distortion = tprob['optical_distortion'],
                # random_fog = tprob['random_fog'],
                # clahe = tprob['clahe'],
                hue_sat_val = tprob['hue_sat_val'],
                random_brightness = tprob['random_brightness'],
                random_gamma = tprob['random_gamma']
                )
        self.transform.createPresetTransform()
        self.transform.createFrameTransform()
        
    # Provjerava postoji li direktorij za izlazne slike
    def checkOutdir(self):
        outdirs = []
        for key, val in gen_samples.items():
            if val:
                outdirs.append(f'{self.outdir}/{key}')
        labels_dir = '%s/labels' %self.outdir
        jpeg_dir = '%s/jpeg' %self.outdir
        
        if path.exists(outdirs[0]):
            print(f'{outdirs[0]} vec postoji')
            self.num_samples['smoke'] = len(glob(f'{outdirs[0]}/s_*.npz'))
            self.num_samples['nosmoke'] = len(glob(f'{outdirs[0]}/ns_*.npz'))
        else:
            os.makedirs(outdirs[0])
            print(f'{outdirs[0]} ne postoji, kreiram novi')

        for outdir in outdirs[1:]:
            if path.exists(outdir):
                print(f'{outdir} vec postoji')
                check_smoke = len(glob(f'{outdir}/s_*.npz'))
                check_nosmoke = len(glob(f'{outdir}/ns_*.npz'))
                # provjeravam slaze li se broj uzoraka
                if check_smoke != self.num_samples['smoke'] or check_nosmoke != self.num_samples['nosmoke']: 
                    sys.exit()
            else:
                print(f'{outdir} ne postoji, kreiram novi')
                os.makedirs(outdir)
                
        if path.exists(labels_dir):
            print(f'{labels_dir} vec postoji')
            check_smoke = len(glob(f'{labels_dir}/s_*.txt'))
            check_nosmoke = len(glob(f'{labels_dir}/ns_*.txt'))
            # provjeravam slaze li se broj uzoraka za slike i labele
            if check_smoke != self.num_samples['smoke'] or check_nosmoke != self.num_samples['nosmoke']: 
                sys.exit()
        else:
            print(f'{labels_dir} ne postoji, kreiram novi')
            os.makedirs(labels_dir)
                
        if path.exists(jpeg_dir):
            print(f'{jpeg_dir} vec postoji')
            '''
            check_smoke = len(glob(f'{jpeg_dir}/s_*.jpg'))
            check_nosmoke = len(glob(f'{jpeg_dir}/ns_*.jpg'))
            # provjeravam slaze li se broj uzoraka za slike i jpegova
            if check_smoke != self.num_samples['smoke'] or check_nosmoke != self.num_samples['nosmoke']: 
                sys.exit()
            '''
        else:
            print(f'{jpeg_dir} ne postoji, kreiram novi')
            os.makedirs(jpeg_dir)

    # Stvara sliku udaljenosti (horizont postavljen na 25500m)
    def getAugDistImg(self):
        pixinfo = self.readPixinfo()
        if pixinfo is None:
            return np.zeros((self.parms['IMG_HEIGHT'],
                             self.parms['IMG_WIDTH']))
        
        pixinfo[pixinfo>DIST_MAX] = DIST_MAX
        pixinfo[pixinfo==GIS_HORIZONT] = DIST_HORIZONT
        pixinfo = np.uint8(255.0*pixinfo/DIST_HORIZONT)
        
        if self.transform is not None:
            # Treba mi maska (prazna) za generirat transformacije
            wmask = np.zeros((self.parms['IMG_HEIGHT'],
                              self.parms['IMG_WIDTH']))
            self.transform.setImage(image=pixinfo, mask=wmask)
            self.transform.applySequenceGeometricTransform()
            pixinfo = self.transform.aug['image']
        
        # Apsolutna udaljenost
        dist = cv2.split(pixinfo)[CH_PIXDIST] 
        return dist

    # Foreground slika (razlika i adaptivni prag)
    def getForegroundImg(self):
        D = np.uint8(self.fgextractor.getDiffImg()*255)
        T = np.uint8(self.fgextractor.getThresholdImg()*255)
        D = np.reshape(D, (self.aug_h, self.aug_w, 1))
        T = np.reshape(T, (self.aug_h, self.aug_w, 1))
        return np.concatenate((D, T), 2)


    # Gradijenti: 1. gradx, grady; 2. magnituda i kut
    def getGradients(self, img):
        # Operate on BLUE channel only
        sobelx = cv2.Sobel(img[:,:,0], cv2.CV_32F, 1, 0)
        sobely = cv2.Sobel(img[:,:,0], cv2.CV_32F, 0, 1)
        
        mag, ang = cv2.cartToPolar(sobelx, sobely)

        gxn = np.uint8(255.0 * (sobelx+SOBEL_MAX)/(2*SOBEL_MAX))
        gyn = np.uint8(255.0 * (sobely+SOBEL_MAX)/(2*SOBEL_MAX))

        magn = np.uint8(255.0 * mag/MAG_MAX)
        angn = np.uint8(255.0 * ang/(2*np.pi))

        grad = cv2.merge([gxn, gyn])
        gpol = cv2.merge([magn, angn])

        return grad, gpol

    # Racuna temporalnu sliku    
    def updateTemporalFrame(self, frindex, first_set, alpha=0.1):
        wframe = self.aug_frame[:,:,0]/255.0 # Only BLUE channel
        if frindex == 0:    # Prvi frame  prolazu
            if first_set:   # Prvi frame u prvom setu, svi na novu sliku
                self.temporal_frame = np.empty((self.aug_h, self.aug_w, 3),
                                               dtype=np.float32)
                for k in range(3):
                    self.temporal_frame[:,:,k] = wframe
                return  # To je to za prvi put
            else:  # Update dugorocne memorije sa prethodnim prolazom
                self.temporal_frame[:,:,0] = self.temporal_frame[:,:,1]
                
        # Updateam sve
        self.temporal_frame[:,:,1] = (1-alpha) * \
                                     self.temporal_frame[:,:,1] + \
                                     alpha*wframe
        self.temporal_frame[:,:,2] = wframe
    

    def getTemporalFrame(self):
        tframe=np.empty((self.aug_h, self.aug_w, 3), dtype=np.uint8)
        tframe[:,:,0] = self.temporal_frame[:,:,0]*255
        tframe[:,:,1] = self.temporal_frame[:,:,1]*255
        tframe[:,:,2] = self.temporal_frame[:,:,2]*255
        return tframe

    # provjerava ima li dima na slici (da, ne, nedefinirano)
    def isSmokeImage(self, gt):
        npix = np.sum(gt==255)
        if npix >= MIN_SMOKE_PIX:
            return True
        elif npix == 0:
            return False
        else:
            return None
    
            
    #
    # Izdvaja set slika s uzorkom dima iz jedne slike sekvence
    #
    def smokeSetCreate(self):
        if not self.isSmokeImage(self.aug_mask):  # Nema dima na slici
            return False
        
        mask = self.aug_mask
        tframe = self.getTemporalFrame()
        #cv2.imshow('TEMP', tframe)
        #key = cv2.waitKey()
        #print(key)
        #if (key == 13):
        #    print('JE JE')
        #    rnum = random.randint(0,1000)
        #    fffr = 'WORK/temp%05d.jpg' %rnum
        #    cv2.imwrite(fffr, tframe)
        #    fffr = 'WORK/still%05d.jpg' %rnum
        #    cv2.imwrite(fffr, self.frame)
        
        
        fginfo = self.getForegroundImg()
        frame = self.aug_frame
        
        # Racunavam gradijente
        fgrad, fgpol = self.getGradients(frame)
        
        dist = self.aug_dist.copy()
        dist = dist.reshape(dist.shape[0], dist.shape[1], 1)
        
        has_smoke = False
        
        rows, cols = frame.shape[0:2] 
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)

        last_x0 = None
        for cnt in contours:
            # Preskoci male povrsine
            if cv2.contourArea(cnt) < MIN_SMOKE_PIX/2:
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            
            # Slucajan pomak prozora u odnosu na konture dima
            dx = ROI_XSIZE - w
            dy = ROI_YSIZE - h
            rfx = int(dx*random.random())
            rfy = int(dy*random.random())
            
            x0 = max(x-rfx, 0)
            y0 = max(y-rfy, 0)
            x1 = min(x0+ROI_XSIZE, cols)
            y1 = min(y0+ROI_YSIZE, rows)
            if (y1-y0) < ROI_YSIZE or (x1-x0) < ROI_XSIZE:
                continue # Nije dovoljno veliki sample, preskoci
            
            if not self.isSmokeImage(mask[y0:y1, x0:x1]):
                continue  # mali dim, preskoci
            # Pronasli smo dim

            if last_x0: # Provjeravam jesmo li vec uzeli koji ROI
                # Jeli ova kontura cijela ukljucena u prethodni ROI?
                if x0>last_x0 and x1<last_x1 and y0>last_y0 and y1<last_y1:
                    # Je! Provjeravam postoji li minimalni pomak
                    move_x = abs(x0-last_x0)  # Pomak po x
                    move_y = abs(y0-last_y0)  # Pkmak po y
                    if move_x < ROI_XSIZE/3 and move_y < ROI_YSIZE/3:
                        # Ne postoji dovoljan pomak, preskacem
                        continue

            # Uzimamo ROI (x0,y0)-(y1,y1)
            has_smoke = True
            # Pamtim lastx0, lasty0 da osiguran da sljdeci ROI
            # ima minimalni dopusteni odmak
            last_x0 = x0
            last_y0 = y0
            last_x1 = x1
            last_y1 = y1
            
            smoke_img = frame[y0:y1, x0:x1]
            smoke_tem = tframe[y0:y1, x0:x1]
            dist_img = dist[y0:y1, x0:x1]
            fg_img = fginfo[y0:y1, x0:x1]
            smoke_gt = mask[y0:y1, x0:x1]

            grad_img = fgrad[y0:y1, x0:x1]
            gpol_img = fgpol[y0:y1, x0:x1]
            
            # Teoretski, moze biti vise od jednog oznacenog dima
            # na ovom uzorku
            mask_contours, _ = cv2.findContours(smoke_gt,
                                                cv2.RETR_EXTERNAL,
                                                cv2.CHAIN_APPROX_SIMPLE)
            l_lines = ''
            
            for maskcnt in mask_contours:
                if cv2.contourArea(cnt) < MIN_SMOKE_PIX/3:
                    continue
                mcx, mcy, mcw, mch = cv2.boundingRect(maskcnt)
                mcx += int(mcw/2.0)
                mcy += int(mch/2.0)
                l_lines += '0 %f %f %f %f\n' %(float(mcx)/ROI_XSIZE,
                                               float(mcy)/ROI_YSIZE,
                                               float(mcw)/ROI_XSIZE,
                                               float(mch)/ROI_YSIZE)
            img_name = 's_%06d' %self.num_samples['smoke']
            f_still =f'{self.outdir}/still/{img_name}.npz' 
            f_tem =f'{self.outdir}/temporal/{img_name}.npz' 
            f_join_st =f'{self.outdir}/join_st/{img_name}.npz' 
            f_dist =f'{self.outdir}/dist/{img_name}.npz' 
            f_join_std =f'{self.outdir}/join_std/{img_name}.npz' 
            f_fgr =f'{self.outdir}/fgr/{img_name}.npz' 
            f_grad =f'{self.outdir}/grad/{img_name}.npz' 
            f_gpolar =f'{self.outdir}/gpolar/{img_name}.npz' 
            f_join_all =f'{self.outdir}/join_all/{img_name}.npz' 
            f_label = f'{self.outdir}/labels/{img_name}.txt'
            f_jpeg = f'{self.outdir}/jpeg/{img_name}.jpg'
            
            if gen_samples['still']:
                np.savez_compressed(f_still, im=smoke_img)
            if gen_samples['temporal']:
                np.savez_compressed(f_tem, im=smoke_tem)
            if gen_samples['join_st']: # Join still, temporal
                conimg = np.concatenate((smoke_img, smoke_tem[:,:,:2]), 2)
                np.savez_compressed(f_join_st, im=conimg)
            if gen_samples['dist']: # Join still, dist
                conimg = np.concatenate((smoke_img, dist_img), 2)
                np.savez_compressed(f_dist, im=conimg)
            if gen_samples['join_std']: # Join still, temp, dist
                conimg = np.concatenate((smoke_img, smoke_tem[:,:,:2],
                                         dist_img), 2)
                np.savez_compressed(f_join_std, im=conimg)
            if gen_samples['fgr']: # Join still, fgr
                conimg = np.concatenate((smoke_img, fg_img), 2)
                np.savez_compressed(f_fgr, im=conimg)
            if gen_samples['grad']: # Join still, gradient
                conimg = np.concatenate((smoke_img, grad_img), 2)
                np.savez_compressed(f_grad, im=conimg)
            if gen_samples['gpolar']: # Join still, gradient
                conimg = np.concatenate((smoke_img, gpol_img), 2)
                np.savez_compressed(f_gpolar, im=conimg)
            if gen_samples['join_all']: # Join all
                    conimg = np.concatenate((smoke_img,
                                             smoke_tem[:,:,:2],
                                             dist_img,
                                             fg_img,
                                             grad_img,
                                             gpol_img), 2)
                    np.savez_compressed(f_join_all, im=conimg)
                    
            # Save jpeg image with smoke bounding boxes
            jpeg_img = smoke_img.copy()
            cv2.drawContours(jpeg_img, mask_contours, -1, (255,0,0))
            cv2.imwrite(f_jpeg, jpeg_img)
            # save labels
            with open(f_label, 'w') as f:
                f.write(l_lines)
            self.num_samples['smoke'] += 1
                                    
        return has_smoke

    #
    # Izdvaja set slika bez dima iz jedne slike sekvence
    #
    def nosmokeSetCreate(self):
        # Ne zelim uzorke na kojima postoji bilo kakva mogucnost dima
        has_smoke = self.isSmokeImage(self.aug_mask)
        if has_smoke is None: # Granicno, ima malo, ali ne dovoljno dima  
            return False
        elif has_smoke == True: # Ima dim
            return False
        
        # Sigurno nema dima, ajmo dalje
        tframe = self.getTemporalFrame()
        fginfo = self.getForegroundImg()
        frame = self.aug_frame
        
        # Racunam gradijente
        fgrad, fgpol = self.getGradients(frame)
        
        dist = self.aug_dist
        dist = dist.reshape(dist.shape[0], dist.shape[1], 1)
        
        rows, cols = frame.shape[:2]

        skip_top = int(min((rows-ROI_YSIZE)/2.0, SKIP_TOP))
        skip_bottom = int(min((rows-ROI_YSIZE)/3.0, SKIP_BOTTOM))
        skip_horizontal = int(min((cols-ROI_XSIZE)/2, SKIP_HORIZONTAL))

        if skip_top > 0:
            ys = int(skip_top/2) + random.randrange(int(skip_top))
        else:
            ys = 0

        if skip_bottom > 0:
            roi_ye = rows - random.randrange(skip_bottom)
        else:
            roi_ye = rows

        if skip_horizontal > 0:
            roi_xe = cols - random.randrange(skip_horizontal)
        else:
            roi_xe = cols
        
        xstride = ROI_XSIZE + random.randrange(int(1.2*ROI_XSIZE))
        ystride = ROI_YSIZE + random.randrange(int(1.2*ROI_YSIZE))
        
                
        ye = int(ys + ROI_YSIZE)
        
        while ye <= roi_ye:
            if skip_horizontal > 0:
                xs = int(random.randrange(skip_horizontal))
            else:
                xs = 0
            xe = xs + ROI_XSIZE
            while xe <= roi_xe:
                img_roi = frame[ys:ye, xs:xe, :]
                tem_roi = tframe[ys:ye, xs:xe, :]
                dist_roi = dist[ys:ye, xs:xe]
                fg_roi = fginfo[ys:ye, xs:xe,:]
                grad_roi = fgrad[ys:ye, xs:xe,:]
                gpol_roi = fgpol[ys:ye, xs:xe,:]
                
                img_name = 'ns_%06d' %self.num_samples['nosmoke']
                f_still =f'{self.outdir}/still/{img_name}.npz' 
                f_tem =f'{self.outdir}/temporal/{img_name}.npz' 
                f_join_st =f'{self.outdir}/join_st/{img_name}.npz' 
                f_dist =f'{self.outdir}/dist/{img_name}.npz' 
                f_join_std =f'{self.outdir}/join_std/{img_name}.npz' 
                f_fgr =f'{self.outdir}/fgr/{img_name}.npz' 
                f_grad =f'{self.outdir}/grad/{img_name}.npz' 
                f_gpolar =f'{self.outdir}/gpolar/{img_name}.npz' 
                f_join_all =f'{self.outdir}/join_all/{img_name}.npz' 
                f_label = f'{self.outdir}/labels/{img_name}.txt'
                f_jpeg = f'{self.outdir}/jpeg/{img_name}.jpg'
                
                if gen_samples['still']:
                    np.savez_compressed(f_still, im=img_roi)
                if gen_samples['temporal']:
                    np.savez_compressed(f_tem, im=tem_roi)
                if gen_samples['join_st']: # Join: still, temporal
                    conimg = np.concatenate((img_roi,
                                             tem_roi[:,:,:2]), 2)
                    np.savez_compressed(f_join_st, im=conimg)
                if gen_samples['dist']: # Join: still, dist
                    conimg = np.concatenate((img_roi, dist_roi), 2)
                    np.savez_compressed(f_dist, im=conimg)
                if gen_samples['join_std']: # Join: still, temp, dist
                    conimg = np.concatenate((img_roi,
                                             tem_roi[:,:,:2],
                                             dist_roi), 2)
                    np.savez_compressed(f_join_std, im=conimg)
                if gen_samples['fgr']: # Join still, fgr
                    conimg = np.concatenate((img_roi, fg_roi), 2)
                    np.savez_compressed(f_fgr, im=conimg)
                if gen_samples['grad']: # Join still, gradient
                    conimg = np.concatenate((img_roi, grad_roi), 2)
                    np.savez_compressed(f_grad, im=conimg)
                if gen_samples['gpolar']: # Join still, grad polar
                    conimg = np.concatenate((img_roi, gpol_roi), 2)
                    np.savez_compressed(f_gpolar, im=conimg)
                # Join still, temp, dist, fgr
                if gen_samples['join_all']: 
                    conimg = np.concatenate((img_roi,
                                             tem_roi[:,:,:2],
                                             dist_roi,
                                             fg_roi,
                                             grad_roi,
                                             gpol_roi), 2)
                    np.savez_compressed(f_join_all, im=conimg)
                    
                # Save jpeg image with smoke bounding boxes
                # cv2.imwrite(f_jpeg, img_roi)
                # Save labels
                with open(f_label, 'w') as f:
                    f.write('')
                self.num_samples['nosmoke'] += 1
                        
                xs += xstride
                xe = xs + ROI_XSIZE
                
            ys += ystride
            ye = ys + ROI_YSIZE
        
    #
    # Izdvaja set uzoraka s dimom i set bez dima
    #
    def extractSet(self):
        start_index = self.parms['START_INDEX']
        end_index = self.parms['END_INDEX']
        max_index = self.parms['MAX_INDEX']
        cur_index = start_index

        rolleup = False
        if start_index > end_index:
            rolleup = True
        
        # Apsolutna udaljenost
        # Ako je postavljena transformacija, generira se
        self.aug_dist = self.getAugDistImg()
        self.aug_h, self.aug_w = self.aug_dist.shape
        
        self.seq_running = True
        while self.seq_running:
            # resetiramo augmentaciju za zaustavljanje u presetu
            if self.transform is not None:
                self.transform.resetPreset()
                
            # Na prvi frame    
            frindex = 0
            skip_frames = 1 + random.randrange(SKIP_FRAMES)
            # slucajni indeks na kojem uzimam no_smoke set
            no_smoke_frindex = SKIP_FRAMES + random.randrange(SKIP_FRAMES)
            while True and self.seq_running:
                self.frame = self.readFrame(cur_index, frindex)
                if self.frame is None:
                    break
                
                self.mask = self.readMask(cur_index, frindex)
                if self.mask is None:
                    self.seq_running = False
                    break

                # Augmentacija (ako je postavljena)
                if self.transform is not None:
                    self.transform.setImage(image=self.frame,
                                            mask=self.mask)
                    # Transformacija koja se primjenjuje na cijelu sekvencu
                    self.transform.applySequenceTransform()
                    # transformacija koja se primjenjuje na ovo zaustavljanje
                    self.transform.applyPresetTransform()
                    # transformacija za svaki frame
                    self.transform.applyFrameTransform()
                    
                    self.aug_frame = self.transform.aug['image']
                    self.aug_mask = self.transform.aug['mask']
                else:
                    self.aug_frame = self.frame
                    self.aug_mask = self.mask
                
                self.updateTemporalFrame(frindex, cur_index==start_index)
                self.fgextractor.apply(self.aug_frame, frindex==0)
                # samo prvi frame u nizu
                if cur_index != start_index:
                    if skip_frames == 0:
                        if self.smokeSetCreate():  # Generirani uzorci dima
                            skip_frames = SKIP_FRAMES # Preskoci frameove
                            self.sequence_has_smoke = True
                    else:
                        skip_frames = max(0, skip_frames-1)
                            
                    if frindex==no_smoke_frindex and not self.sequence_has_smoke:
                        self.nosmokeSetCreate()
                        
                frindex += 1 
                del self.frame, self.mask
                del self.aug_frame, self.aug_mask
                
            cur_index += 1
            if cur_index > max_index:
                cur_index = 1
                rolleup = False
            if rolleup==False and cur_index>end_index:
                self.seq_running = False


            
        
# Direktno startanje iz komandne linije
AUX_TRANS_PROB = 0.2

transformations = {'scale': 1.0,
                   'random_scale': AUX_TRANS_PROB,
                   'horizontal_flip': AUX_TRANS_PROB,
                   'rotate': AUX_TRANS_PROB,
                   'perspective': AUX_TRANS_PROB,
                   'optical_distortion': AUX_TRANS_PROB,
                   # 'random_fog': AUX_TRANS_PROB,
                   # 'clahe': AUX_TRANS_PROB,
                   'hue_sat_val': AUX_TRANS_PROB,
                   'random_brightness': AUX_TRANS_PROB,
                   'random_gamma': AUX_TRANS_PROB
                   }
'''
transformations = {'random_scale': 0,
                   'horizontal_flip': 0,
                   'rotate': 0,
                   'perspective': 0,
                   'optical_distortion': 0,
                   'random_fog': 0,
                   'clahe': 0,
                   'hue_sat_val': 0,
                   'random_brightness': 0,
                   'random_gamma': 0
                   }
'''                   
                   
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Koristenje: %s -i <input_dir> -o <output_dir> [-r -l -h]'
              %sys.argv[0])
        exit()
        
    seq_dir = None
    outdir = None
    reduced_set = False # Za test i valid idu samo originalne slike
    limited_set = False # Ogranicena augmentacija
    
    try:
        opts, args = getopt.getopt(sys.argv[1:], "hrli:o:",
                                   ["input=","output="])
    except getopt.GetoptError:
        print('Koristenje: %s -i <input_dir> -o <output_dir> [-r -l -h]'
              %sys.argv[0])
        sys.exit(2)
        
    for opt, arg in opts:
        if opt == '-h':
            print('Koristenje: %s -i <input_dir> -o <output_dir> [-r -h]'
                  %sys.argv[0])
            sys.exit()
        elif opt in ("-i", "--input"):
            seq_dir = arg
        elif opt in ("-o", "--output"):
            outdir = arg
        elif opt in ("-r", "--reduced"):
            reduced_set = True # Samo originalne slike, bez augmentacije
        elif opt in ("-l", "--limited"):
            limited_set = True # Limitirana augmentacija
            
    if seq_dir is None or outdir is None:
        print('Koristenje: %s -i <input_dir> -o <output_dir> [-h]'
              %sys.argv[0])
        sys.exit()

    print('SEKVENCA: %s' %path.basename(seq_dir))
    seq = AugmentedYoloSet(seq_dir=seq_dir, outdir=outdir)
    if not seq.checkParms():
        print(f'{seq_dir}: no DET_PARMS.txt ... skipping')
    elif not seq.checkPixinfo():
        print(f'{seq_dir}: no pixinfo.yml ... skipping')
    else:
        seq.checkOutdir()
        # Bez transformacija - original (1)
        seq.extractSet()
        print('Original\n\tS: %5d\tNS: %6d'
              %(seq.num_samples['smoke'], seq.num_samples['nosmoke']))

        ####################################################
        # Transformacije - iskljuciti za TEST i VALIDATION
        #
        # Ne radim za test i validation
        if reduced_set == True:
            sys.exit()

        # Horizontal Flip (2)
        transformations = {'scale': 1.0,
                           'random_scale': AUX_TRANS_PROB,
                           'horizontal_flip': 1,
                           'rotate': AUX_TRANS_PROB,
                           'perspective': AUX_TRANS_PROB,
                           'optical_distortion': AUX_TRANS_PROB,
                           'hue_sat_val': AUX_TRANS_PROB,
                           'random_brightness': AUX_TRANS_PROB,
                           'random_gamma':AUX_TRANS_PROB 
                           }
        seq = AugmentedYoloSet(seq_dir=seq_dir, outdir=outdir)
        seq.checkParms()
        seq.checkOutdir()
        seq.setAugmentation(transformations)
        seq.extractSet()
        print('Horizontal Flip\n\tS: %5d\tNS: %6d'
              %(seq.num_samples['smoke'], seq.num_samples['nosmoke']))
        
        # Up Scale (3) - Povecaj x2
        transformations = {'scale': 2.0,
                           'random_scale': 0,
                           'horizontal_flip': AUX_TRANS_PROB,
                           'rotate': AUX_TRANS_PROB,
                           'perspective': AUX_TRANS_PROB,
                           'optical_distortion': AUX_TRANS_PROB,
                           'hue_sat_val': AUX_TRANS_PROB,
                           'random_brightness': AUX_TRANS_PROB,
                           'random_gamma': AUX_TRANS_PROB
                           }
        seq = AugmentedYoloSet(seq_dir=seq_dir, outdir=outdir)
        seq.checkParms()
        seq.checkOutdir()
        seq.setAugmentation(transformations)
        seq.extractSet()
        print('Up scale\n\tS: %5d\tNS: %6d'
              %(seq.num_samples['smoke'], seq.num_samples['nosmoke']))
        
        ####################################################
        # Preostale transformacije ne koristim za
        # limited augmentation
        # 
        if limited_set == True:
            sys.exit()
            
        #
        # Down Scale (4) - Smanji maksimalno
        transformations = {'scale': 0.6,
                           'random_scale': 0,
                           'horizontal_flip': AUX_TRANS_PROB,
                           'rotate': AUX_TRANS_PROB,
                           'perspective': AUX_TRANS_PROB,
                           'optical_distortion': AUX_TRANS_PROB,
                           'hue_sat_val': AUX_TRANS_PROB,
                           'random_brightness': AUX_TRANS_PROB,
                           'random_gamma': AUX_TRANS_PROB
                           }
        seq = AugmentedYoloSet(seq_dir=seq_dir, outdir=outdir)
        seq.checkParms()
        seq.checkOutdir()
        seq.setAugmentation(transformations)
        seq.extractSet()
        print('Down scale\n\tS: %5d\tNS: %6d'
              %(seq.num_samples['smoke'], seq.num_samples['nosmoke']))

        # Rotate (5)
        transformations = {'scale': 1.0,
                           'random_scale': AUX_TRANS_PROB,
                           'horizontal_flip': AUX_TRANS_PROB,
                           'rotate': 1,
                           'perspective': AUX_TRANS_PROB,
                           'optical_distortion': AUX_TRANS_PROB,
                           'hue_sat_val': AUX_TRANS_PROB,
                           'random_brightness': AUX_TRANS_PROB,
                           'random_gamma': AUX_TRANS_PROB
                           }
        seq = AugmentedYoloSet(seq_dir=seq_dir, outdir=outdir)
        seq.checkParms()
        seq.checkOutdir()
        seq.setAugmentation(transformations)
        seq.extractSet()
        print('Rotate\n\tS: %5d\tNS: %6d'
              %(seq.num_samples['smoke'], seq.num_samples['nosmoke']))

        # Optical Distortion (6)
        transformations = {'scale': 1.0,
                           'random_scale': AUX_TRANS_PROB,
                           'horizontal_flip': AUX_TRANS_PROB,
                           'rotate': AUX_TRANS_PROB,
                           'perspective': AUX_TRANS_PROB,
                           'optical_distortion': 1,
                           'hue_sat_val': AUX_TRANS_PROB,
                           'random_brightness': AUX_TRANS_PROB,
                           'random_gamma': AUX_TRANS_PROB
                           }
        seq = AugmentedYoloSet(seq_dir=seq_dir, outdir=outdir)
        seq.checkParms()
        seq.checkOutdir()
        seq.setAugmentation(transformations)
        seq.extractSet()
        print('Optical distortion\n\tS: %5d\tNS: %6d'
              %(seq.num_samples['smoke'], seq.num_samples['nosmoke']))

        # Perspective (7)
        transformations = {'scale': 1.0,
                           'random_scale': AUX_TRANS_PROB,
                           'horizontal_flip': AUX_TRANS_PROB,
                           'rotate': AUX_TRANS_PROB,
                           'perspective': 1,
                           'optical_distortion': AUX_TRANS_PROB,
                           'hue_sat_val': AUX_TRANS_PROB,
                           'random_brightness': AUX_TRANS_PROB,
                           'random_gamma': AUX_TRANS_PROB
                           }
        seq = AugmentedYoloSet(seq_dir=seq_dir, outdir=outdir)
        seq.checkParms()
        seq.checkOutdir()
        seq.setAugmentation(transformations)
        seq.extractSet()
        print('Perspective\n\tS: %5d\tNS: %6d'
              %(seq.num_samples['smoke'], seq.num_samples['nosmoke']))

        # Colors (8)
        transformations = {'scale': 1.0,
                           'random_scale': AUX_TRANS_PROB,
                           'horizontal_flip': AUX_TRANS_PROB,
                           'rotate': AUX_TRANS_PROB,
                           'perspective': AUX_TRANS_PROB,
                           'optical_distortion': AUX_TRANS_PROB,
                           'hue_sat_val': 0.6,
                           'random_brightness': 1,
                           'random_gamma': 0.6
                           }
        seq = AugmentedYoloSet(seq_dir=seq_dir, outdir=outdir)
        seq.checkParms()
        seq.checkOutdir()
        seq.setAugmentation(transformations)
        seq.extractSet()
        print('Color\n\tS: %5d\tNS: %6d'
              %(seq.num_samples['smoke'], seq.num_samples['nosmoke']))
        
       
        
                
