from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
import os
from PIL import Image
import numpy as np
import itertools
import random
import math

def load_image(path):
    return Image.open(path).convert('RGB')

#--------------------------------------------------------------------------------------

class RotateTransformer(object):
#data augmentation transformer     #input data transformer
    def __init__(self, crop, flip):
        super(RotateTransformer, self).__init__()
        self.crop = crop
        self.flip = flip
        self.angles = [0, 90, 180, 270]

    def __call__(self, img, rot=None):
        img = TF.resize(img, (256, 256))
        img = TF.crop(img, self.crop[0], self.crop[1], 224, 224)
        if self.flip:
            img = TF.hflip(img)
        if rot is not None:
            img = TF.rotate(img, self.angles[rot])
        img = TF.to_tensor(img)
        img = TF.normalize(img, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        return img
    
#--------------------------------------------------------------------------------------

class JP_Transformer(object):

    def __init__(self,grid_num,permutation_num):
        super(JP_Transformer, self).__init__()
        self.grid_num=grid_num
        self.permutations = np.array(list(itertools.permutations(list(range(grid_num)),grid_num)))
        temp=np.arange(len(self.permutations))
        step=len(self.permutations)//permutation_num
        self.permutations=self.permutations[temp%step==0]

    def __call__(self, img, permutation):
        img = TF.resize(img, (224, 224))
        #img = TF.crop(img, (224, 224))
        #img = TF.crop(img, self.crop[0], self.crop[1], 224, 224)
        if (permutation is not None):
            tiles=[]
            tile_height =tile_width =img.size[0]/math.sqrt(self.grid_num)
            currentx = 0
            currenty = 0
            while currenty < img.size[1]:
                while currentx < img.size[0]:
                    tile = img.crop((currentx,currenty,currentx + tile_width,currenty + tile_height))
                    tiles.append(tile)
                    currentx += tile_width
                currenty += tile_height
                currentx = 0
            tiles = [tiles[self.permutations[permutation][t]] for t in range(len(tiles))]
            currentx = 0
            currenty = 0
            t=0
            while currenty < img.size[1]:
                while currentx < img.size[0]:
                    tile = img.paste(tiles[t],(int(currentx),int(currenty)))
                    currentx += tile_width
                    t+=1
                currenty += tile_height
                currentx = 0
        img = TF.to_tensor(img)
        img = TF.normalize(img, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        return img
    
#--------------------------------------------------------------------------------------
def get_relative_rotation(rgb_rot, depth_rot):
    rel_rot = rgb_rot - depth_rot
    if rel_rot < 0:
        rel_rot += 4
    assert rel_rot in range(4)
    return rel_rot
#--------------------------------------------------------------------------------------
def get_relative_jig(rgb_rot, depth_rot):
    rel_rot = abs(rgb_rot - depth_rot)
    
    return rel_rot

#--------------------------------------------------------------------------------------
class DatasetGenerator(Dataset):
    def __init__(self, data_path, data_txt, mode,dtype=None,do_rot=False, JP=False,augmentation=False,grid_num=4,permutation_num=24,transform=None):
        self.data_path = data_path
        self.data_txt = data_txt
        self.augmentation=augmentation
        self.JP = JP
        self.do_rot=do_rot
        self.grid_num=grid_num
        self.permutation_num=permutation_num
        self.dtype=dtype
        self.Path_rgb=[]
        self.Path_depth=[]
        self.Target=[]
        self.transform=transform
        self.mode = mode
        with open(self.data_txt) as f:
            line = f.readline()
            while line:
                temp = line.strip().split(' ')
                img_path=temp[0]
                img_label=temp[1]
                if self.dtype=='ROD':
                    img_path=os.path.join(data_path,"ROD\\"+img_path)
                    path_rgb = img_path.replace('***', 'crop')
                    path_rgb = path_rgb.replace('???', 'rgb')
                    path_depth = img_path.replace('***', 'depthcrop')
                    path_depth = path_depth.replace('???', 'surfnorm')
                if self.dtype=='synROD':
                    img_path=os.path.join(data_path,img_path)
                    if 'depth' in img_path:
                        path_rgb = img_path.replace('depth', 'rgb')+".png"
                        path_depth = img_path+".png"
                self.Path_rgb.append(path_rgb)
                self.Path_depth.append(path_depth)
                self.Target.append(int(img_label))
                line = f.readline()
        
    def __getitem__(self, index):
        
        #print(index,self.Path_rgb[index],self.Path_depth[index])
        img_rgb = load_image(self.Path_rgb[index])
        img_depth = load_image(self.Path_depth[index])
        target=self.Target[index]
        if self.mode=='rotation':
            rot_rgb = None
            rot_depth = None
            if self.transform is not None:
                img_rgb = self.transform(img_rgb)
                img_depth = self.transform(img_depth)
            else:  # Otherwise define a random one (random cropping, random horizontal flip)
                top = random.randint(0, 256 - 224)
                left = random.randint(0, 256 - 224)
                flip = random.choice([True, False])
                if self.do_rot:
                    rot_rgb = random.choice([0, 1, 2, 3])
                    rot_depth = random.choice([0, 1, 2, 3])

                transform = RotateTransformer([top, left], flip)
                # Apply the same transform to both modalities, rotating them if required
                img_rgb = transform(img_rgb, rot_rgb)
                img_depth = transform(img_depth, rot_depth)
                
            if self.do_rot and (self.transform is None):
                return img_rgb, img_depth, target, get_relative_rotation(rot_rgb, rot_depth)
            return img_rgb, img_depth, target

        elif self.mode=='jigsaw_puzzle':
            jp_rgb = None
            jp_depth = None
            if self.JP:
                jp_rgb = random.choice(range(self.permutation_num))
                jp_depth = random.choice(range(self.permutation_num))

            transform = JP_Transformer(grid_num=self.grid_num,permutation_num=self.permutation_num)
            
            img_rgb = transform(img_rgb, jp_rgb)
            img_depth = transform(img_depth, jp_depth)

            if self.JP :
                return img_rgb, img_depth, target, get_relative_jig(jp_rgb, jp_depth)
            return img_rgb, img_depth, target
        else:
            raise Exception("unknown mode!!")

    def __len__(self):
        return len(self.Target)