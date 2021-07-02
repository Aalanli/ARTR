# %%
import json
import os
import random

import numpy as np
import torch
import torchvision
import torchvision.transforms.functional as F
from torch.utils.data import Dataset

from PIL import Image
from pycocotools import mask as coco_mask
from matplotlib import pyplot as plt


class CocoBoxes(Dataset):
    def __init__(self, image_dir: str, json_file: str, query_transforms=None, transforms=None, case4: bool=False, case4_dir: str=None) -> None:
        """
        Dataset class which gives bounding boxes, query images respective input images
        for arbitrary bounding box detection following four cases.

        Case1: One query image is matched to its bounding box, from the same image.

        Case2: Multiple query images are matched to multiple bounding boxes, from the same image.

        Case3: Multiple query images are matched to all the bounding boxes, all sampled from the same image.

        Case4: Some query images are pulled from the same class as the bounding boxes, all bounding boxes of
               image belonging to that class is given.

        For cases 1 to 3, all query images must have their own bounding box, but not all bounding
        boxes needs to have a direct matching to one of the query images.

        Args:
            case4: whether to activative case4 or not, if True, all query images must be written to disk for efficiency;
                   a new directory must be constructed.
            case4_dir: if case4 is True, the location of such a directory
        """

        self.image_dir = image_dir
        self.json_file = json_file
        self.transforms = transforms
        self.query_transforms = query_transforms
        self.case4 = case4

        with open(self.json_file, 'r') as f:
            annote = json.load(f)
        
        self.image_ids = {}  # key = image_ids, item = nested dict classes and their bounding boxes
        for i in annote['annotations']:
            if i['image_id'] in self.image_ids:
                if i['category_id'] in self.image_ids[i['image_id']]:
                    self.image_ids[i['image_id']][i['category_id']].append(i['bbox'])
                else:
                    self.image_ids[i['image_id']].update({i['category_id']: [i['bbox']]})
            else:
                self.image_ids[i['image_id']] = {i['category_id']: [i['bbox']]}
        self.keys = list(self.image_ids.keys())
        del annote

    def id_to_file(self, id: str) -> str:
        """image_id to file name"""
        id = str(id)
        file = '0' * (12 - len(id)) + id + '.jpg'
        return self.image_dir + '/' + file
    
    def random_query_bbox(self, bboxes: np.ndarray):
        """
        For cases 1 to 3
        """
        q = len(bboxes)
        indx = np.random.permutation(q)
        q_stop = np.random.randint(low=1, high=q, size=1)
        b_stop = np.random.randint(low=q_stop, high=q, size=1)
        return bboxes[indx[:q_stop]], bboxes[indx[:b_stop]]  # query images, bboxes; query images <= bboxes
    
    def fetch_query(self, image, bbox):
        """
        bbox = x, y of top most point, x right, y down from that point
        """

        if type(image) == str:
            img = Image.open(image, mode='RGB')
        return F.crop(img, bbox[1], bbox[0], bbox[3], bbox[2])
    
    def __len__(self):
        return len(self.keys)
    
    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()
        
        img_path = self.id_to_file(self.keys[index])
        bboxes = self.image_ids(self.keys[index])
        img_class: int = random.choices(list(bboxes.keys()), k=1)[0]  # randomly chose an image class to preform detection

        img = Image.open(img_path, 'RBG')  # the target image
        bboxes = np.asarray(bboxes[img_class])  # all the boxes in that class belonging to the target image
        if self.transforms is not None:
            img, bboxes = self.transforms(img, bboxes)
        
        # TODO: make case4 work
        queries, bboxes = self.random_query_bbox(bboxes)
        queries = self.fetch_query(img, queries)
        if self.query_transforms is not None:
            queries = self.query_transforms(queries)
        
        return img, queries, bboxes


class CocoDetection(torchvision.datasets.CocoDetection):
    def __init__(self, img_folder, ann_file, transforms, return_masks):
        super(CocoDetection, self).__init__(img_folder, ann_file)
        self._transforms = transforms
        self.prepare = ConvertCocoPolysToMask(return_masks)

    def __getitem__(self, idx):
        img, target = super(CocoDetection, self).__getitem__(idx)
        image_id = self.ids[idx]
        target = {'image_id': image_id, 'annotations': target}
        img, target = self.prepare(img, target)
        if self._transforms is not None:
            img, target = self._transforms(img, target)
        return img, target

def convert_coco_poly_to_mask(segmentations, height, width):
    masks = []
    for polygons in segmentations:
        rles = coco_mask.frPyObjects(polygons, height, width)
        mask = coco_mask.decode(rles)
        if len(mask.shape) < 3:
            mask = mask[..., None]
        mask = torch.as_tensor(mask, dtype=torch.uint8)
        mask = mask.any(dim=2)
        masks.append(mask)
    if masks:
        masks = torch.stack(masks, dim=0)
    else:
        masks = torch.zeros((0, height, width), dtype=torch.uint8)
    return masks


class ConvertCocoPolysToMask(object):
    def __init__(self, return_masks=False):
        self.return_masks = return_masks

    def __call__(self, image, target):
        w, h = image.size

        image_id = target["image_id"]
        image_id = torch.tensor([image_id])

        anno = target["annotations"]

        anno = [obj for obj in anno if 'iscrowd' not in obj or obj['iscrowd'] == 0]

        boxes = [obj["bbox"] for obj in anno]
        # guard against no boxes via resizing
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        classes = [obj["category_id"] for obj in anno]
        classes = torch.tensor(classes, dtype=torch.int64)

        if self.return_masks:
            segmentations = [obj["segmentation"] for obj in anno]
            masks = convert_coco_poly_to_mask(segmentations, h, w)

        keypoints = None
        if anno and "keypoints" in anno[0]:
            keypoints = [obj["keypoints"] for obj in anno]
            keypoints = torch.as_tensor(keypoints, dtype=torch.float32)
            num_keypoints = keypoints.shape[0]
            if num_keypoints:
                keypoints = keypoints.view(num_keypoints, -1, 3)

        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        classes = classes[keep]
        if self.return_masks:
            masks = masks[keep]
        if keypoints is not None:
            keypoints = keypoints[keep]

        target = {}
        target["boxes"] = boxes
        target["labels"] = classes
        if self.return_masks:
            target["masks"] = masks
        target["image_id"] = image_id
        if keypoints is not None:
            target["keypoints"] = keypoints

        # for conversion to coco api
        area = torch.tensor([obj["area"] for obj in anno])
        iscrowd = torch.tensor([obj["iscrowd"] if "iscrowd" in obj else 0 for obj in anno])
        target["area"] = area[keep]
        target["iscrowd"] = iscrowd[keep]

        target["orig_size"] = torch.as_tensor([int(h), int(w)])
        target["size"] = torch.as_tensor([int(h), int(w)])

        return image, target
