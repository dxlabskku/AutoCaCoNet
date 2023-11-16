import os

import cv2
from matplotlib import pyplot as plt
import tensorflow as tf
from keras.models import load_model

from utils import load_image_test

norm = lambda x:(x / 127.5) - 1
rev_norm1 = lambda x:(x + 1) * 127.5
rev_norm = lambda x: x * 0.5 + 0.5

PATH = 'data'

def test():
    test_dataset = tf.data.Dataset.list_files(PATH+'/test/*.jpg')
    test_dataset = test_dataset.map(load_image_test)
    test_dataset = test_dataset.batch(1)
    
    model = load_model('AnimeColorizationModelv5')
    
    for img in test_dataset:
        test_img = model.predict(x=norm(img))
        plt.imsave(PATH+"/predicted_test/"+str(i)+".png", rev_norm(test_img)[0])
    
if __name__=='__main__':
    test()
