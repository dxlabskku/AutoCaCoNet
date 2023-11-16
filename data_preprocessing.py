import cv2
import os
import numpy as np

class Preprossesor :
    def __init__(self):
        self.dir_path = './data/'
        raw_data_path = os.path.join(self.dir_path, 'raw/')
        path_list = os.listdir(raw_data_path)
        
        path_list.sort()
        self.path_list = [os.path.join(raw_data_path, x) for x in path_list]

        print(self.path_list)

        self.resized_data_path = os.path.join(self.dir_path, 'color')
        if not os.path.exists(self.resized_data_path):
            os.makedirs(self.resized_data_path)

        self.edged_data_path = os.path.join(self.dir_path, 'line-drawing')
        if not os.path.exists(self.edged_data_path):
            os.makedirs(self.edged_data_path)
            
        self.train_data_path = os.path.join(self.dir_path, 'train')
        if not os.path.exists(self.train_data_path):
            os.makedirs(self.train_data_path)


    def preprocess_save(self):
        for path in self.path_list:
            img = cv2.imread(path)
            img_resized = cv2.resize(img[75:-75, :, :], (256, 256),interpolation = cv2.INTER_LINEAR)
            mask = np.mean(img, axis=2) < 80
            outline = (cv2.Canny(img, 100, 180) / 255).astype(np.bool)
            outline = np.logical_not(np.logical_or(outline, mask)).astype(np.uint8) * 255
            outline = cv2.resize(outline[75:-75, :], (256, 256),interpolation = cv2.INTER_AREA)
           
            file_name = path.split('/')[-1]
            resized_path = os.path.join(self.resized_data_path, file_name)
            edged_path = os.path.join(self.edged_data_path, file_name)
            train_path = os.path.join(self.train_data_path, file_name)
            
            cv2.imwrite(resized_path, img_resized)
            cv2.imwrite(edged_path, outline)
            
            line = cv2.imread(edged_path)
            color = cv2.imread(resized_path)
            train = cv2.hconcat([line, color])
            
            cv2.imwrite(train_path, train)
            
            
def main():
    P = Preprossesor()
    P.preprocess_save()
    
if __name__=='__main__':
    main()
