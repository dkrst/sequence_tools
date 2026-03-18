import cv2
import numpy as np
import albumentations as A
from matplotlib import pyplot as plt
import random

AUX_TRANS_PROB = 0.2
PRESET_TRANS_PROB = 0.4
FRAME_TRANS_PROB = 0.5

class AlbumentationsProcessor():
    def __init__(self):
        self.sequenceGeoParms = None
        self.sequencePixelparms = None
        self.presetParms = None
        self.orig = None
        self.work = None
        self.aug = None
        self.scale = 1.0
        
    def setImage(self, image, mask):
        if self.scale > 1.0:   # Up Scale
            simage = cv2.resize(image, None,
                                fx=self.scale, fy=self.scale,
                                interpolation=cv2.INTER_CUBIC)
            smask = cv2.resize(mask, None,
                                fx=self.scale, fy=self.scale,
                                interpolation=cv2.INTER_NEAREST)
        elif self.scale < 1.0: # Down Scale
            simage = cv2.resize(image, None,
                                fx=self.scale, fy=self.scale,
                                interpolation=cv2.INTER_AREA)
            smask = cv2.resize(mask, None,
                                fx=self.scale, fy=self.scale,
                                interpolation=cv2.INTER_AREA)
        else:                  # Original size
            simage = image
            smask = mask

        self.orig = {'image': simage, 'mask': smask}
        self.work = {'image': simage, 'mask': smask}
        self.aug = None

    def resetSequence(self):
        self.sequenceGeoParms = None
        self.sequencePixParms = None

    def resetPreset(self):
        self.presetParms = None
        
    def getAugmentation(self):
        return self.aug

    def getOrig(self):
        return self.orig
        
    #
    # SequenceTransform
    #
    # Transformacije koje se primjenjuju na cijelu sekvencu
    #    - Geometrijske: primjenjuju se i na pixel mape
    #    - Pixel: primjenjuju se samo na piksele (sadrzaj) slike
    #
    def createSequenceTransform(self,
                                # Geometrijske
                                scale=1.0,
                                random_scale=AUX_TRANS_PROB,
                                horizontal_flip=AUX_TRANS_PROB,
                                rotate=AUX_TRANS_PROB,
                                perspective=AUX_TRANS_PROB,
                                optical_distortion = AUX_TRANS_PROB,
                                # Pixel level (glavne)
                                hue_sat_val = AUX_TRANS_PROB,
                                random_brightness = AUX_TRANS_PROB,
                                random_gamma = AUX_TRANS_PROB,
                                # Pixel level (pomocne)
                                random_fog=AUX_TRANS_PROB,
                                clahe = AUX_TRANS_PROB,
                                # 
                                fancy_pca=AUX_TRANS_PROB,
                                multiplicative_noise = AUX_TRANS_PROB,
                                sharpen = AUX_TRANS_PROB,
                                advanced_blur = AUX_TRANS_PROB
                                ):
        
        if scale != 1.0 and scale > 0:  # Global zoom
            self.scale = scale
            
        self.createSequenceGeometric(random_scale = random_scale,
                                     horizontal_flip = horizontal_flip,
                                     rotate = rotate,
                                     perspective = perspective,
                                     optical_distortion = optical_distortion
                                     )
        self.createSequencePixel(random_fog = random_fog,
                                 clahe = clahe,
                                 fancy_pca = fancy_pca,
                                 multiplicative_noise = multiplicative_noise,
                                 hue_sat_val = hue_sat_val,
                                 random_brightness = random_brightness,
                                 random_gamma = random_gamma,
                                 sharpen = sharpen,
                                 advanced_blur = advanced_blur
                                 )
        
    #
    # SequenceGeometric
    #
    # Geometrijske transformacije
    #    - primjenjuje se na cijelu sekvencu
    #    - primjenjuju se na pixel mape
    #
    def createSequenceGeometric(self,
                                perspective, random_scale,
                                rotate, horizontal_flip,
                                optical_distortion
                                ):
        self.sequence_geometric =  A.ReplayCompose([
            # Perspective
            A.Perspective (scale=(0.1, 0.3), keep_size=True,
                           border_mode=cv2.BORDER_CONSTANT,
                           fit_output=False,
                           interpolation=cv2.INTER_CUBIC,
                           p=perspective),
            # RandomScale
            A.RandomScale (scale_limit=(-0.3, 1),
                           interpolation=cv2.INTER_CUBIC,
                           p=random_scale),
            # Rotate
            A.Rotate (limit=(-15, 15), interpolation=cv2.INTER_CUBIC,
                      border_mode=cv2.BORDER_CONSTANT,
                      rotate_method='largest_box',
                      crop_border=False, p=rotate),
            # HorizontalFlip
            A.HorizontalFlip(p=horizontal_flip),
            # OpticalDistortion
            A.OpticalDistortion (distort_limit=(-0.4, 0.4),
                                 #shift_limit=(-0.2, 0.2),
                                 interpolation=cv2.INTER_CUBIC,
                                 #border_mode=cv2.BORDER_CONSTANT,
                                 p=optical_distortion),

        ])
        self.sequenceGeoParms = None

        
    #
    # SequencePixel
    #
    # Pixel transformacije (mijenjaju izgled slike, ne geometriju)
    #    - primjenjuje se na cijelu sekvencu
    #
    def createSequencePixel(self,
                            # Glavne
                            random_fog, clahe,
                            # Pomocne
                            fancy_pca, multiplicative_noise,
                            hue_sat_val, random_brightness,
                            random_gamma, sharpen, advanced_blur
                            ):
        self.sequence_pixel =  A.ReplayCompose([
            #
            # Glavne
            #
            # HSV
            A.HueSaturationValue (hue_shift_limit=9, sat_shift_limit=9,
                                  val_shift_limit=30, p=hue_sat_val),
            #RabdomBrightnessContrast
            A.RandomBrightnessContrast(brightness_limit=(-0.2, 0.2),
                                       contrast_limit=(-0.2, 0.2),
                                       brightness_by_max=True,
                                       p=random_brightness),
            #RandomGamma
            A.RandomGamma(gamma_limit=(60, 150), p=random_gamma),
            #
            # POMOCNE
            #
            # RandomFog
            A.RandomFog(fog_coef_range=(0.05, 0.15),
                        alpha_coef=0.1, p=random_fog),
            # Clahe
            A.CLAHE (clip_limit=[1.0, 1.5], tile_grid_size=(8, 8), p=clahe),
            # FancyPCA
            A.FancyPCA (alpha=0.3, p=fancy_pca),
            # MultiplicativeNoise
            A.MultiplicativeNoise (multiplier=(0.9, 1.1),
                                   per_channel=True, elementwise=False,
                                   p=multiplicative_noise),
            # Sharpen
            A.Sharpen (alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=sharpen),
            # AdvancedBlur
            A.AdvancedBlur (blur_limit=(3, 7), #sigma_x_limit=(0.2, 1.0),
                            #sigma_y_limit=(0.2, 1.0),
                            #sigmaX_limit=None, sigmaY_limit=None,
                            rotate_limit=90, beta_limit=(0.5, 8.0),
                            noise_limit=(0.9, 1.1),
                            p=advanced_blur) 

        ])
        self.sequencePixParms = None

        
    #
    # PresetTransform - primjenjuje se na svako zaustavljanje, ne dira masku
    def createPresetTransform(self, 
                              random_fog = PRESET_TRANS_PROB,
                              #iso_noise = PRESET_TRANS_PROB
    ):
        self.preset_transform =  A.ReplayCompose([
            # RandomFog
            A.RandomFog(fog_coef_range=(0.05, 0.15),
                        alpha_coef=0.1, p=random_fog),
            #A.ISONoise (color_shift=(0.01, 0.05), intensity=(0.05, 0.2),
            #            p=iso_noise)
        ])
        self.presetParms = None
        
    #
    # FrameTransform - primjenjuje se na svaki frame, ne dira masku
    def createFrameTransform(self, 
                             iso_noise = FRAME_TRANS_PROB
    ):
        self.frame_transform =  A.Compose([
            A.ISONoise (color_shift=(0.01, 0.05), intensity=(0.05, 0.1),
                        p=iso_noise)            
        ])
        

    ########

    #
    # SequenceGeometricTransform - na cijelu sekvencu, mijenja maske
    def applySequenceGeometricTransform(self):
        if self.work == None:
            raise ValueError('No input data')

        if self.sequenceGeoParms == None:
            geo_data = self.sequence_geometric(image=self.work['image'],
                                               mask=self.work['mask'])
            self.sequenceGeoParms = geo_data['replay']
        else:
            geo_data = A.ReplayCompose.replay(self.sequenceGeoParms,
                                              image=self.work['image'],
                                              mask=self.work['mask'])
            
        self.aug = {'image': geo_data['image'], 'mask': geo_data['mask']}
        self.work = {'image': geo_data['image'], 'mask': geo_data['mask']}
            

    #
    # SequencePixelTransform - na cijelu sekvencu, ne mijenja maske
    def applySequencePixelTransform(self):
        if self.work == None:
            raise ValueError('No input data')

        if self.sequencePixParms == None:
            pix_data = self.sequence_pixel(image=self.work['image'])
            self.sequencePixParms = pix_data['replay']
        else:
            pix_data = A.ReplayCompose.replay(self.sequencePixParms,
                                              image=self.work['image'])
             
        self.aug = {'image': pix_data['image'], 'mask': self.work['mask']}
        self.work = {'image': pix_data['image'], 'mask': self.work['mask']}
 
    #
    # SequenceTransform - na cijelu sekvencu, mijenja maske
    def applySequenceTransform(self):
        if self.work == None:
            raise ValueError('No input data')

        self.applySequenceGeometricTransform()
        self.applySequencePixelTransform()

        
    #
    # PresetTransform - na svako zaustavljanje, ne mijenja masku
    def applyPresetTransform(self):
        if self.work == None:
            raise ValueError('No input data')

        if self.presetParms == None:
            aug_data = self.preset_transform(image=self.work['image'])
            self.presetParms = aug_data['replay']
        else:
            aug_data = A.ReplayCompose.replay(self.presetParms,
                                              image=self.work['image'])

        self.aug['image'] = aug_data['image']
        self.work['image'] = aug_data['image']
        
    #
    # FrameTransform - na svaki frame, ne mijenja masku
    def applyFrameTransform(self):
        #pass
        if self.work == None:
            raise ValueError('No input data')

        aug_data = self.frame_transform(image=self.work['image'])
        self.aug['image'] = aug_data['image']
        self.work['image'] = aug_data['image']
        
