import random
import cv2
from dataset import PASCALVOC
import numpy as np
import torch
from torchvision.transforms import functional as F

class ChangeExposure(object):
    def __init__(self, prob):
        self.prob = prob
    
    def __call__(self, image, target):
        if random.random() < self.prob:
            image_hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            h, s, v = cv2.split(image_hsv)
            adjust = random.uniform(- 1 / 3, 1 / 2)
            v = v * (1 + adjust)
            v = np.clip(v, 0, 255).astype(image_hsv.dtype)
            image_hsv = cv2.merge((h,s,v))
            image = cv2.cvtColor(image_hsv, cv2.COLOR_HSV2RGB)
        return image, target

class ChangeSaturation(object):
    def __init__(self, prob):
        self.prob = prob
    
    def __call__(self, image, target):
        if random.random() < self.prob:
            image_hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            h, s, v = cv2.split(image_hsv)
            adjust = random.uniform(- 1 / 3, 1 / 2)
            s = s * (1 + adjust)
            s = np.clip(s, 0, 255).astype(image_hsv.dtype)
            image_hsv = cv2.merge((h,s,v))
            image = cv2.cvtColor(image_hsv, cv2.COLOR_HSV2RGB)
        return image, target

class RandomScale(object):
    def __init__(self, prob):
        self.prob = prob
    
    def __call__(self, image, target):
        if random.random() < self.prob:
            scale_w = random.uniform(0.8, 1.2)
            scale_h = random.uniform(0.8, 1.2)
            h, w, c = image.shape
            image = cv2.resize(image, (int(w * scale_w), int(h * scale_h)))
        return image, target

class Resize(object):
    def __init__(self, width, height):
        self.width = width
        self.height = height
    
    def __call__(self, image, target):
        image = cv2.resize(image, (self.width, self.height))
        return image, target

class RandomTranslation(object):
    def __init__(self, prob):
        self.prob = prob
    
    def __call__(self, image, target):
        if random.random() < self.prob:
            h, w, c = image.shape
            ratio_x = random.uniform(-0.2, 0.2)
            ratio_y = random.uniform(-0.2, 0.2)
            trans_x = int(ratio_x * w)
            trans_y = int(ratio_y * h)
            result = np.zeros((h, w, c), dtype=image.dtype)
            
            if trans_x >= 0 and trans_y >= 0:
                result[trans_y:, trans_x:, :] = image[:h - trans_y, :w - trans_x, :]
            elif trans_x < 0 and trans_y >= 0:
                result[trans_y:, :w + trans_x, :] = image[:h - trans_y, -trans_x:, :] # Note trans_x is < 0!
            elif trans_x >= 0 and trans_y < 0:
                result[:h + trans_y, trans_x:, :] = image[-trans_y:, :w - trans_x, :]
            elif trans_x < 0 and trans_y < 0:
                result[:h + trans_y, :w + trans_x, :] = image[-trans_y:, -trans_x:, :]
            
            result_target = torch.zeros(7, 7, 30)
            for i in range(7):
                for j in range(7):
                    if target[i, j, 4].item() == 1:
                        # Note the center is x first and y second, so it's the reverse of i, j.
                        center = target[i, j, 0:2] * (1 / 7) + torch.tensor([j * 1 / 7, i * 1 / 7])
                        image_left_corner = center  - torch.tensor([target[i, j, 2] / 2, target[i, j, 3] / 2])
                        image_left_corner = image_left_corner + torch.tensor([ratio_x, ratio_y])
                        image_right_corner = center + torch.tensor([target[i, j, 2] / 2, target[i, j, 3] / 2])
                        image_right_corner = image_right_corner + torch.tensor([ratio_x, ratio_y])
                        if image_left_corner[0] >= 1 or image_right_corner[0] <= 0 or image_left_corner[1] >= 1 or image_right_corner[1] <= 0:
                            result_target[i, j, :] = 0
                        image_left_corner = torch.clamp(image_left_corner, 0, 1)
                        image_right_corner = torch.clamp(image_right_corner, 0, 1)
                        
                        center = (image_left_corner + image_right_corner) / 2
                        widthandheight = image_right_corner - image_left_corner
                        
                        grid = torch.ceil(center*7) - 1
                        grid[grid < 0] = 0
                        grid_i = int(grid[1])
                        grid_j = int(grid[0])

                        left_corner = torch.tensor([grid_j * 1 / 7, grid_i * 1 / 7])
                        relative_center = (center - left_corner) / (1 / 7)

                        result_target[grid_i, grid_j, 0:2] = relative_center
                        result_target[grid_i, grid_j, 2:4] = widthandheight
                        result_target[grid_i, grid_j, 4] = 1

                        result_target[grid_i, grid_j, 5:7] = relative_center
                        result_target[grid_i, grid_j, 7:9] = widthandheight
                        result_target[grid_i, grid_j, 9] = 1

                        result_target[grid_i, grid_j, 10:] = target[i, j, 10:]
        else:
            result = image
            result_target = target    
        return result, result_target

class ToTensor(object):
    def __call__(self, image, target):
        image = F.to_tensor(image)
        return image, target

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target