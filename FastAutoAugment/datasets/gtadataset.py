from zipfile import ZipFile
from PIL import Image, ImageFile
import numpy as np
import os
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from FastAutoAugment.transforms_target import  PILToLongTensor
ImageFile.LOAD_TRUNCATED_IMAGES = True
IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)
NUM_CLASSES = 19
# colour map
label_colours = [[128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156],
                 [190, 153, 153], [153, 153, 153], [250, 170, 30], [220, 220, 0],
                 [107, 142, 35], [152, 251, 152], [0, 130, 180], [220, 20, 60],
                 [255, 0, 0], [0, 0, 142],
                 [0, 0, 70], [0, 60, 100], [0, 80, 100],
                 [0, 0, 230], [119, 11, 32], [0, 0, 0]]  # the color of ignored label(-1)
label_colours = list(map(tuple, label_colours))

name_classes = ['road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 'trafflight', 'traffsign', 'vegetation',

                'terrain', 'sky', 'person', 'rider',
                'car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle', 'unlabeled']


def decode_labels(mask, num_images=1, num_classes=NUM_CLASSES):
    """Decode batch of segmentation masks.

    Args:
      mask: result of inference after taking argmax.
      num_images: number of images to decode from the batch.
      num_classes: number of classes to predict.

    Returns:
      A batch with num_images RGB images of the same size as the input.
    """
    if isinstance(mask, torch.Tensor):
        mask = mask.data.cpu().numpy()
    n, h, w = mask.shape
    # if n < num_images:
    num_images = n
    outputs = np.zeros((num_images, h, w, 3), dtype=np.uint8)
    for i in range(num_images):
        img = Image.new('RGB', (len(mask[i, 0]), len(mask[i])))
        pixels = img.load()
        for j_, j in enumerate(mask[i, :, :]):
            for k_, k in enumerate(j):
                if k < num_classes:
                    pixels[k_, j_] = label_colours[k]
        outputs[i] = np.array(img)
    return torch.from_numpy(outputs.transpose([0, 3, 1, 2]).astype('float32')).div_(255.0)


def inspect_decode_labels(pred, num_images=1, num_classes=NUM_CLASSES,
                          inspect_split=[0.9, 0.8, 0.7, 0.5, 0.0], inspect_ratio=[1.0, 0.8, 0.6, 0.3]):
    """Decode batch of segmentation masks accroding to the prediction probability.
    Args:
      pred: result of inference.
      num_images: number of images to decode from the batch.
      num_classes: number of classes to predict (including background).
      inspect_split: probability between different split has different brightness.

    Returns:
      A batch with num_images RGB images of the same size as the input.
    """
    if isinstance(pred, torch.Tensor):
        pred = pred.data.cpu().numpy()
    n, c, h, w = pred.shape
    pred = pred.transpose([0, 2, 3, 1])
    if n < num_images:
        num_images = n
    outputs = np.zeros((num_images, h, w, 3), dtype=np.uint8)
    for i in range(num_images):
        img = Image.new('RGB', (w, h))
        pixels = img.load()
        for j_, j in enumerate(pred[i, :, :, :]):
            for k_, k in enumerate(j):
                assert k.shape[0] == num_classes
                k_value = np.max(softmax(k))
                k_class = np.argmax(k)
                for it, iv in enumerate(inspect_split):
                    if k_value > iv: break
                if iv > 0:
                    pixels[k_, j_] = tuple(map(lambda x: int(inspect_ratio[it] * x), label_colours[k_class]))
        outputs[i] = np.array(img)
    return torch.from_numpy(outputs.transpose([0, 3, 1, 2]).astype('float32')).div_(255.0)


class GTA5_Dataset(data.Dataset):
    def __init__(self, data_root_path='/content/GTA5', split='train', transform_pre=None,
                 transform_after=None, transform_target_after=None, seed=1, sample='full'):
        assert os.path.exists(data_root_path)
        self.img_path_zip, self.gt_path_zip = os.path.join(data_root_path, 'images.zip'), os.path.join(data_root_path,
                                                                                                       'labels.zip')
        assert (os.path.exists(self.img_path_zip) and os.path.exists(self.gt_path_zip))

        self.split = split
        item_list_filepath = os.path.join(data_root_path, self.split + ".txt")
        assert os.path.exists(item_list_filepath)
        self.img_archive, self.image_filepath = self.open_zip(self.img_path_zip)
        self.gt_archive, self.gt_filepath = self.open_zip(self.gt_path_zip)
        self.items = np.array([int(id.strip()) for id in open(item_list_filepath)])
        if sample == 'full':
            pass
        else:
            rang = np.random.RandomState(seed)
            self.items = np.random.choice(self.items, sample, replace=False)
        # self.id_to_trainid = {7: 0,8: 1,11: 2,12: 3,13: 4,17: 5,19: 6,20: 7,21: 8,22: 9,23: 10,24: 11,25: 12,26: 13,27: 14,28: 15,31: 16,32: 17,33: 18,34: 13}
        self.id_to_trainid = d = {7: 0, 8: 1, 11: 2, 12: 3, 13: 4, 17: 5, 18: 5, 19: 6, 20: 7, 21: 8, 22: 9, 23: 10,
                                  24: 11, 25: 12, 26: 13, 34: 13, 27: 14, 28: 15, 31: 16,
                                  32: 17, 33: 18, 0: 19, 1: 19, 2: 19, 3: 19, 4: 19, 5: 19, 6: 19, 9: 19, 10: 19,
                                  14: 19, 15: 19, 16: 19, 29: 19, 30: 19}

        self.transform_after = transform_after if transform_after is not None else transforms.ToTensor()
        self.transform_target_after = transform_target_after if transform_target_after is not None  else PILToLongTensor()
        self.transfom_pre = transform_pre if transform_pre  is not None else None

        print("{} num images in GTA5 {} set have been loaded.".format(len(self.items), self.split))

    @property
    def targets(self):
        return self.items

    def __getitem__(self, item):
        id = self.items[item]
        image = Image.open(self.img_archive.open(os.path.join(self.image_filepath, "{:0>5d}.png".format(id)))).convert(
            "RGB")
        gt = Image.open(self.gt_archive.open(os.path.join(self.gt_filepath, "{:0>5d}.png".format(id))))
        gt = self._mask_transform(gt)

        if self.transfom_pre is not None :
            image, gt = self.transfom_pre(image ,gt )
        image = self.transform_after(image)
        gt = self.transform_target_after(gt)

        return image, gt

    def _mask_transform(self, gt_image):  # check if cross entropy ask from long or float
        target = np.array(gt_image, np.uint8)
        for k, v in self.id_to_trainid.items():
            target[target == k] = v
        return Image.fromarray(target)

    def open_zip(self, filename):
        archive = ZipFile(filename)
        entries = archive.infolist()
        entries_filename = [e.filename for e in entries]
        return archive, entries_filename[0]

    def __len__(self):
        return self.items.shape[0]

# '''
# 7 -- road                 	 [128.  64, 128]  7 : 0
# 8 -- sidewalk             	 [244.  35, 232]  8 : 1
# 11 -- building             	 [70. 70, 70]     11 : 2
# 12 -- wall                 	 [102. 102, 156]  12 : 3
# 13 -- fence                	 [190. 153, 153]  13 : 4
# 17 -- pole                 	 [153. 153, 153]  17 : 5
# 18 -- polegroup            	 [153. 153, 153]  18 :5
# 19 -- traffic light        	 [250. 170,  30]  19 : 6
# 20 -- traffic sign         	 [220. 220,   0]  20 : 7
# 21 -- vegetation           	 [107. 142,  35]  21 : 8
# 22 -- terrain              	 [152. 251, 152]  22 : 9
# 23 -- sky                  	 [ 70. 130, 180]  23 : 10
# 24 -- person               	 [220.  20,  60]  24 : 11
# 25 -- rider                	 [255.   0,   0]  25 : 12
# 26 -- car                  	 [  0.   0, 142]  26 : 13
# 34 -- license plate        	 [  0.   0, 142]  34 : 13
# 27 -- truck                	 [ 0.  0, 70]     27 : 14
# 28 -- bus                  	 [  0.  60, 100]  28 : 15
# 31 -- train                	 [  0.  80, 100]  31 : 16
# 32 -- motorcycle           	 [  0.   0, 230]  32 : 17
# 33 -- bicycle              	 [119.  11,  32]  33 : 18
# 0 -- unlabeled            	 [0. 0, 0]        0 :  18
# 1 -- ego vehicle          	 [0. 0, 0]        1 :  18
# 2 -- rectification border 	 [0. 0, 0]        2 :  18
# 3 -- out of roi           	 [0. 0, 0]        3 :  18
# 4 -- static               	 [20. 20, 20]     4 :  18
# 5 -- dynamic              	 [111.  74,   0]  5 :  18
# 6 -- ground               	 [81.  0, 81]     6 :  18
# 9 -- parking              	 [250. 170, 160]  9 :  18
# 10 -- rail track           	 [230. 150, 140]  10 : 18
# 14 -- guard rail           	 [180. 165, 180]  14 : 18
# 15 -- bridge               	 [150. 100, 100]  15 : 18
# 16 -- tunnel               	 [150. 120,  90]  16 : 18
# 29 -- caravan              	 [ 0.  0, 90]     29 : 18
# 30 -- trailer              	 [  0.   0, 110]  30 : 18
# --------------------------------------
# '''
#
# d = {7: 0, 8: 1, 11: 2, 12: 3, 13: 4, 17: 5, 18: 5, 19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12, 26: 13, 34: 13, 27: 14, 28: 15, 31: 16,
#      32: 17, 33: 18, 0: 19, 1: 19, 2: 19, 3: 19, 4: 19, 5: 19, 6: 19, 9: 19, 10: 19, 14: 19, 15: 19, 16: 19, 29: 19, 30: 19}
