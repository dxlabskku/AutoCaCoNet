import glob

import cv2
import numpy as np
from matplotlib import pyplot as plt

from linefiller.trappedball_fill import trapped_ball_fill_multi, flood_fill_multi, mark_fill, build_fill_map, merge_fill, show_fill_map
from linefiller.thinning import thinning

EDGE_PATH = 'data/line-drawing/'
PATH = 'data/predicted_test/'

def linefiller(img):
    ret, binary = cv2.threshold(im, 155, 255, cv2.THRESH_BINARY)
    
    fills = []
    result = binary
    
    fill = trapped_ball_fill_multi(result, 3, method='max')
    fills += fill
    result = mark_fill(result, fill)
    
    fill = trapped_ball_fill_multi(result, 2, method=None)
    fills += fill
    result = mark_fill(result, fill)
    
    fill = trapped_ball_fill_multi(result, 1, method=None)
    fills += fill
    result = mark_fill(result, fill)


    fill = flood_fill_multi(result)
    fills += fill

    fillmap = build_fill_map(result, fills)
    fillmap

    fillmap = merge_fill(fillmap)
    return fillmap

def colorfiller(fillmap, img):
    r = []

    for o in range(256):
        for p in range(256):
            v = fillmap[o][p]
            r.append(v)

    r = list(set(r))
    r.sort()

    xy_2 = []

    for k in r:
        xy_1=[]
        for l in range(256):
            for m in range(256):
                if fillmap[l][m] == k:
                    xy_1.append((l, m))
        xy_2.append(xy_1)
    
    for h in range(len(xy_2)):
        im_b=[]
        im_g=[]
        im_r=[]

        for n in xy_2[h]:
            im_b.append(im_test[n][0])
            im_g.append(im_test[n][1])
            im_r.append(im_test[n][2])

        b_mean = round(np.mean(im_b))
        g_mean = round(np.mean(im_g))
        r_mean = round(np.mean(im_r))

        for n in xy_2[h]:

            im_test[n][0] = b_mean*(1-(np.abs(im_test[n][0] - b_mean) / 255))
            im_test[n][1] = g_mean*(1-(np.abs(im_test[n][1] - g_mean) / 255))
            im_test[n][2] = r_mean*(1-(np.abs(im_test[n][2] - r_mean) / 255))

def main():
    
    img_paths = sorted(glob.glob(os.path.join(PATH, '*.jpg')))
    edge_paths = sorted(glob.glob(os.path.join(EDGE_PATH, '*.jpg')))

    for edge_path, img_path in zip(edge_paths, img_paths):
        edge_img = cv2.imread(edge_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.imread(img_path)
        fillmap = ilnefiller(edge_img)
        processed_img = colorfiller(fillmap, img)
        file_name = img_path.split('/')[-1]
        cv2.imwrite(os.path.join('processed/', file_name), processed_img)
        
if __name__=='__main__':
    main()
