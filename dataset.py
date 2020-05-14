import torch
import os
import xml.etree.ElementTree as ET
import cv2

VOC_CLASSES = ('aeroplane', 'bicycle', 'bird', 'boat',
               'bottle', 'bus', 'car', 'cat', 
               'chair', 'cow', 'diningtable', 'dog', 
               'horse', 'motorbike', 'person', 'pottedplant',
               'sheep', 'sofa', 'train', 'tvmonitor')

class PASCALVOC(torch.utils.data.Dataset):
    
    #Input: root path and transforms(optional)
    #Output: images(numpy ndarray) target(tensor 7*7*30)
    
    def __init__(self, root, transforms=None):
        self.root = root
        self.transforms = transforms
        
        self.imgs = sorted(os.listdir(os.path.join(self.root, 'JPEGImages')))
        self.annotations = sorted(os.listdir(os.path.join(self.root, 'Annotations')))
        
    def __getitem__(self, idx):
        img_path = os.path.join(self.root, 'JPEGImages', self.imgs[idx])
        annotation_path = os.path.join(self.root, 'Annotations', self.annotations[idx])
        
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w, c = image.shape 
        
        tree = ET.parse(annotation_path)
        objects = tree.findall('object')
        class_index = []
        bndbox_list = []
        for obj in objects:
            name = obj.find('name').text
            bndbox = obj.find('bndbox')
            class_index.append(VOC_CLASSES.index(name))
            bndbox_list.append([float(bndbox.find('xmin').text), 
                                float(bndbox.find('ymin').text), 
                                float(bndbox.find('xmax').text), 
                                float(bndbox.find('ymax').text)])
        
        bndbox_list =  torch.tensor(bndbox_list) / torch.tensor([w, h, w, h])
        
        target = self.target_encoder(class_index, bndbox_list)
        
        if self.transforms is not None:
            image, target = self.transforms(image, target)
        
        return image, target
    
    def __len__(self):
        return len(self.imgs)
    
    def target_encoder(self, class_index, bndbox_list):
        target = torch.zeros(7, 7, 30)
        center = (bndbox_list[:, :2] + bndbox_list[:, 2:]) / 2
        widthandheight = bndbox_list[:, 2:] - bndbox_list[:, :2]
        
        for i in range(len(class_index)):
            grid = torch.ceil(center[i]*7) - 1
            grid[grid < 0] = 0
            grid_i = int(grid[1])
            grid_j = int(grid[0])
            
            left_corner = torch.tensor([grid_j * 1 / 7, grid_i * 1 / 7])
            relative_center = (center[i] - left_corner) / (1 / 7)
            
            target[grid_i, grid_j, 0:2] = relative_center
            target[grid_i, grid_j, 2:4] = widthandheight[i]
            target[grid_i, grid_j, 4] = 1
            
            target[grid_i, grid_j, 5:7] = relative_center
            target[grid_i, grid_j, 7:9] = widthandheight[i]
            target[grid_i, grid_j, 9] = 1
            
            target[grid_i, grid_j, class_index[i]+10] = 1
        
        return target