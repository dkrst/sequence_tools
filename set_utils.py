
#
# VAZNO:
# Sve slike (still, temporal, ..., join_all) se mogu rekonstruirati
# iz join_all slike
#
# KANALI:
# still(3):temporal(2):dist(1):fg(2):grad(2):gpolar(2)
# 

import matplotlib.pyplot as plt
import numpy as np
from glob import glob
import os
import os.path as path
import string
import random
import shutil
import cv2

from sampleProcessor import SampleProcessor

gen_samples = {
    'jpeg': True,          # jpeg image (Still)
    'still': True,         # Still (original) image      3 CH(BGR)
    'temporal': True,      # Temporal image              3 CH
    'join_st': True,       # Still:Temporal              5 CH
    'dist': True,          # Still:Dist                  4 CH
    'join_std': True,      # Still:Temporal:Dist         6 CH
    'fgr': True,           # Still:Fgr                   5 CH
    'join_stfg': False,    # Still:Temporal:Fgr          7 CH
    'join_stfgd': True,    # Still:Temporal:Dist:Fgr     8 CH
    'grad': False,         # Still:Grad                  5 CH
    'gpolar': False,       # Still:GradPolar             5 CH
    'join_all': False      # All                        12 CH
}

def processNoSmoke(indir=None, outdir=None, n1=0, n2=0):
    if not indir or not outdir or n1==0:
        print("Bad input parameters")
        return
    
    if not path.exists(outdir):
        os.makedirs('%s/still' %outdir)
        os.makedirs('%s/temporal' %outdir)
        os.makedirs('%s/labels' %outdir)
        if n2>0:
            os.makedirs('%s/add/still/ns2' %outdir)
            os.makedirs('%s/add/temporal/ns2' %outdir)
            os.makedirs('%s/add/labels/ns2' %outdir)
    
    ns_files = glob('%s/still/ns_*.jpg' %indir)
    random.shuffle(ns_files)
    
    cnt = 0
    wout_dir = outdir
    for src_img in ns_files:
        base_img = path.basename(src_img)
        base_label = base_img.replace('.jpg', '.txt')
        src_tem = '%s/temporal/%s' %(indir, base_img)
        src_label = '%s/labels/%s' %(indir, base_label)
        
        dst_img = '%s/still/%06d_%s' %(wout_dir, cnt, base_img)
        dst_tem = '%s/temporal/%06d_%s' %(wout_dir, cnt, base_img)
        dst_label = '%s/labels/%06d_%s' %(wout_dir, cnt, base_label)
        
        shutil.copy(src_img, dst_img)
        shutil.copy(src_tem, dst_tem)
        shutil.copy(src_label, dst_label)
                
        cnt += 1
        if cnt == n1:
            wout_dir = '%s/add' %outdir
            
        if cnt >= n1+n2:
            break
    

def genOutSamples(indir, outdir, cnt, base_img):
    src_file = f'{indir}/join_all/{base_img}'
    src_img = np.load(src_file)['im']
    h, w = src_img.shape[:2]

    if gen_samples['still']:            # 3C
        dst_file = '%s/still/%06d_%s' %(outdir, cnt, base_img)
        np.savez_compressed(dst_file, im=src_img[:,:,:3])
    if gen_samples['temporal']:         # 3C
        dst_file = '%s/temporal/%06d_%s' %(outdir, cnt, base_img)
        im1 = np.reshape(src_img[:,:,0], (h, w, 1))
        conimg = np.concatenate((im1, src_img[:,:,3:5]), 2)
        np.savez_compressed(dst_file, im=conimg)
    if gen_samples['join_st']:          # 5C
        dst_file = '%s/join_st/%06d_%s' %(outdir, cnt, base_img)
        np.savez_compressed(dst_file, im=src_img[:,:,:5])
    if gen_samples['dist']:             # 4C
        dst_file = '%s/dist/%06d_%s' %(outdir, cnt, base_img)
        im2 = np.reshape(src_img[:,:,5], (h, w, 1))
        conimg = np.concatenate((src_img[:,:,:3], im2), 2)
        np.savez_compressed(dst_file, im=conimg)
    if gen_samples['join_std']:          # 6C
        dst_file = '%s/join_std/%06d_%s' %(outdir, cnt, base_img)
        np.savez_compressed(dst_file, im=src_img[:,:,:6])
    if gen_samples['fgr']:               # 5C
        dst_file = '%s/fgr/%06d_%s' %(outdir, cnt, base_img)
        conimg = np.concatenate((src_img[:,:,:3], src_img[:,:,6:8]), 2)
        np.savez_compressed(dst_file, im=conimg)
    if gen_samples['join_stfg']:          # 7C
        dst_file = '%s/join_stfg/%06d_%s' %(outdir, cnt, base_img)
        conimg = np.concatenate((src_img[:,:,:5], src_img[:,:,6:8]), 2)
        np.savez_compressed(dst_file, im=conimg)
    if gen_samples['join_stfgd']:          # 8C
        dst_file = '%s/join_stfgd/%06d_%s' %(outdir, cnt, base_img)
        np.savez_compressed(dst_file, im=src_img[:,:,:8])
    if gen_samples['grad']:               # 5C
        dst_file = '%s/grad/%06d_%s' %(outdir, cnt, base_img)
        conimg = np.concatenate((src_img[:,:,:3], src_img[:,:,8:10]), 2)
        np.savez_compressed(dst_file, im=conimg)
    if gen_samples['gpolar']:               # 5C
        dst_file = '%s/gpolar/%06d_%s' %(outdir, cnt, base_img)
        conimg = np.concatenate((src_img[:,:,:3], src_img[:,:,10:12]), 2)
        np.savez_compressed(dst_file, im=conimg)
    if gen_samples['join_all']:               # 12C
        dst_file = '%s/join_all/%06d_%s' %(outdir, cnt, base_img)
        np.savez_compressed(dst_file, im=src_img)
    if gen_samples['jpeg']:               # 12C
        dst_work = '%s/jpeg/%06d_%s' %(outdir, cnt, base_img)
        dst_file = path.splitext(dst_work)[0] + '.jpg'
        cv2.imwrite(dst_file, src_img[:,:,:3])
    
    
def processDir(indir=None, outdir=None, val_part=0.0,
               show_samples=False, bf=1.0, move=False):
    if not indir or not outdir:
        print("False indir ili outdir")
        return
    
    if path.exists(outdir):
        print(f'{outdir} already exists')
        return
    
    split_dir = False
    last_dir = None
    if val_part > 0 and val_part < 1:
        train_dir = '%s/train' %outdir
        val_dir = '%s/valid' %outdir
        os.makedirs(f'{train_dir}/labels')
        os.makedirs(f'{val_dir}/labels')
        split_dir = True
    else:
        os.makedirs('%s/labels' %outdir)
        
    for key, val in gen_samples.items():
        if val:
            if split_dir:
                os.makedirs(f'{train_dir}/{key}')
                os.makedirs(f'{val_dir}/{key}')
            else:
                os.makedirs(f'{outdir}/{key}')
            last_dir = key
                
    if last_dir is None:
        print(f'No samples to generate: {gen_samples}')
        return
        
    # smoke_files = glob(f'{indir}/{last_dir}/s_*.npz')
    smoke_files = glob(f'{indir}/join_all/s_*.npz')
    random.shuffle(smoke_files)
            
    cnt = 0
    for src_img in smoke_files:
        base_img = path.basename(src_img)
        base_label = base_img.replace('.npz', '.txt')

        if split_dir:
            if random.random() < val_part:
                wout_dir = val_dir
            else:
                wout_dir = train_dir
        else:
            wout_dir = outdir

        genOutSamples(indir, wout_dir, cnt, base_img)

        if move:
            os.remove(src_img)

        src_label = f'{indir}/labels/{base_label}'
        dst_label = '%s/labels/%06d_%s' %(wout_dir, cnt, base_label)
        if move:
            shutil.move(src_label, dst_label)
        else:
            shutil.copy(src_label, dst_label)

        cnt += 1

        '''
        for key, val in gen_samples.items():
            if val:
                src_img = f'{indir}/{key}/{base_img}' 
                dst_img = '%s/%s/%06d_%s' %(wout_dir, key, cnt, base_img)
                if move:
                    shutil.move(src_img, dst_img)
                else:
                    shutil.copy(src_img, dst_img)
        '''
        

    n_smoke = cnt
    print('Processed smoke images: ', cnt)
    
    if bf>0: 
        n_ns = int(bf * n_smoke)
    else: # bez kopiranja no smoke uzoraka
        return
    
    ns_files = glob(f'{indir}/join_all/ns_*.npz')
    random.shuffle(ns_files)
    
    cnt = 0
    for src_img in ns_files:
        base_img = path.basename(src_img)
        base_label = base_img.replace('.npz', '.txt')
        
        if split_dir:
            if random.random() < val_part:
                wout_dir = val_dir
            else:
                wout_dir = train_dir
        else:
            wout_dir = outdir

        genOutSamples(indir, wout_dir, cnt, base_img)

        if move:
            os.remove(src_img)
        '''
        for key, val in gen_samples.items():
            if val:
                src_img = f'{indir}/{key}/{base_img}' 
                dst_img = '%s/%s/%06d_%s' %(wout_dir, key, cnt, base_img)
                if move:
                    shutil.move(src_img, dst_img)
                else:
                    shutil.copy(src_img, dst_img)
        '''
        src_label = f'{indir}/labels/{base_label}'
        dst_label = '%s/labels/%06d_%s' %(wout_dir, cnt, base_label)
        if move:
            shutil.move(src_label, dst_label)
        else:
            shutil.copy(src_label, dst_label)
                
        cnt += 1
        
        if cnt >= n_ns:
            break
        
    print('Processed no smoke images: ', cnt)

def clearEmptySmoke(indir, outdir, cnt_s=0, cnt_ns=0):
    # smoke images
    img_files = glob('%s/smoke/*_s*.jpg' %indir)
    cnt = cnt_s
    for src_img in img_files:
        src_label = src_img.replace('jpg', 'txt')
        statinfo = os.stat(src_label)
        if statinfo.st_size > 3:    # mora bit bar 5 brojeva
            dst_img = '%s/%06d_s.jpg' %(outdir, cnt)
            dst_label = dst_img.replace('jpg', 'txt')
            
            shutil.copy(src_img, dst_img)
            shutil.copy(src_label, dst_label)
            cnt += 1
            
    n_smoke_imgs = cnt
    
    # nosmoke images
    img_files = glob('%s/no_smoke/*_ns*.jpg' %indir)
    cnt = cnt_ns
    for src_img in img_files:
        src_label = src_img.replace('jpg', 'txt')
        statinfo = os.stat(src_label)
        if statinfo.st_size == 0:    # mora bit prazna
            dst_img = '%s/%06d_ns.jpg' %(outdir, cnt)
            dst_label = dst_img.replace('jpg', 'txt')
            
            shutil.copy(src_img, dst_img)
            shutil.copy(src_label, dst_label)
            cnt += 1
            if cnt >= n_smoke_imgs:
                break

def createImgListFile(indir, filename='images.txt'):
    filepath = '%s/%s'%(indir, filename)
    img_files = glob('%s/*.jpg' %indir)
    random.shuffle(img_files)
    
    f = open(filepath, 'w')
    for img in img_files:
        f.write('%s\n' %img)

    f.close()
        

def balanceSamples(unbalanced_dir):
    tmp_dir = '%s-TMP'%unbalanced_dir
    additional_dir = '%s_additional'%unbalanced_dir
    shutil.move(unbalanced_dir, tmp_dir)
    os.makedirs(unbalanced_dir)
    os.makedirs(additional_dir)
    
    smoke_files = glob('%s/s_*.jpg' %tmp_dir)
    random.shuffle(smoke_files)
    
    num_smoke = len(smoke_files)
    cnt = 0
    for src_img in smoke_files:
        shutil.copy(src_img, '%s/s_%06d.jpg'%(unbalanced_dir, cnt))
        cnt += 1
    print('%6d/%6d uzoraka dima procesirano' %(cnt, num_smoke))
    
    ns_files = glob('%s/ns_*.jpg' %tmp_dir)
    random.shuffle(ns_files)
    num_ns = len(ns_files)
    cnt = 0

    out_dir = unbalanced_dir
    for src_img in ns_files:
        shutil.copy(src_img, '%s/ns_%06d.jpg'%(out_dir, cnt))
        cnt += 1
        if cnt >= num_smoke and out_dir==unbalanced_dir:
            out_dir = additional_dir
            print('%6d/%6d uzoraka bez dima u izvorni direktorij'
                  %(cnt, num_ns))
        if cnt >= 2*num_smoke:
            print('%6d/%6d uzoraka bez dima ukupno procesirano'
                  %(cnt, num_ns))
            break
        
    shutil.rmtree(tmp_dir)
