# %%
import json
import os
import random
import math

import numpy as np
import torch
import torchvision
import torchvision.transforms.functional as F
from torch.utils.data import Dataset

from PIL import Image
from pycocotools import mask as coco_mask
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle

import data.transforms as T


def format_json(json_file):
    """Formats coco json by grouping features by their image_ids"""
    with open(json_file, 'r') as f:
        annote = json.load(f)
    
    image_ids = {}  # key = image_ids, item = nested dict classes and their bounding boxes
    for i in annote['annotations']:
        if i['image_id'] in image_ids:
            if i['category_id'] in image_ids[i['image_id']]:
                image_ids[i['image_id']][i['category_id']].append(i['bbox'])
            else:
                image_ids[i['image_id']].update({i['category_id']: [i['bbox']]})
        else:
            image_ids[i['image_id']] = {i['category_id']: [i['bbox']]}
    return image_ids


def fetch_query(image, bbox):
    """
    Fetches query images, alias to cropping
    Args:
        image: str or PIL.Image
        bbox: coco annotation format; x, y of top most point, x right, y down from that point
    """
    if type(image) == str:
        image = Image.open(image, mode='RGB')
    return F.crop(image, bbox[1], bbox[0], bbox[3], bbox[2])


def id_to_file(id: str, image_dir) -> str:
    """image_id to file name"""
    id = str(id)
    file = '0' * (12 - len(id)) + id + '.jpg'
    return image_dir + '/' + file


def make_query_pool(image_dir, json_file, name):
    if not os.path.exists(name):
        os.mkdir(name)
    image_ids = format_json(json_file)
    classes_counter = {}
    for id in image_ids:
        file = id_to_file(id, image_dir)
        img = Image.open(file)
        for class_id in image_ids[id]:
            path = name + '/' + str(class_id)
            if not os.path.exists(path):
                os.mkdir(path)
                classes_counter[int(class_id)] = 0
            for bbox in image_ids[id][class_id]:
                query: Image.Image = fetch_query(img, bbox)
                query.save(path + '/' + str(classes_counter[class_id]) + '.jpg')
                classes_counter[class_id] += 1
    with open(name + '/instances.json', 'w') as f:
        json.dump(classes_counter, f)
    print('done')


def plot_box(image: np.ndarray, bbox) -> None:
    bbox = np.asarray(bbox)
    plt.imshow(image)
    ax = plt.gca()
    if len(bbox.shape) == 1:
        rect = Rectangle(bbox[:2], bbox[2], bbox[3], linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
    else:
        for box in bbox:
            rect = Rectangle(box[:2], box[2], box[3], linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
    plt.show()


def visualize_coco(i, q, b):
    plot_box(i, b)
    for i in q:
        plt.imshow(i)
        plt.show()


class CocoBoxes(Dataset):
    def __init__(self, image_dir: str, json_file: str, query_transforms=None, transforms=None, query_pool: str=None, case4_prob=0.5, case4_sigma=3) -> None:
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
            query_pool: whether to activative case4 or not, if not None, indicates the directory of the query
                        pool, which is constructed by the make_query_pool function.
            
            case4_prob: the probability of entering case4 if query_pool is not None.

            case4_sigma: the stddev of the amount of images to draw from the query pool
        """

        self.image_dir = image_dir
        self.json_file = json_file
        self.transforms = transforms
        self.query_transforms = query_transforms
        self.query_pool = query_pool
        self.case4_prob = case4_prob
        self.case4_sigma = case4_sigma

        self.image_ids = format_json(json_file)
        self.keys = list(self.image_ids.keys())

        if query_pool is not None:
            with open(os.path.join(query_pool, 'instances.json'), 'r') as f:
                # query instances describes range of file names under each class
                # to prevent loading all file paths into memory
                self.query_instances = json.load(f)
        
        print('{json_file} loaded into memory.')
    
    def random_query_bbox(self, bboxes: np.ndarray):
        """
        For cases 1 to 3
        """
        q = len(bboxes)
        indx = np.random.permutation(q)
        q_stop = random.randint(1, q)
        b_stop = random.randint(q_stop, q)
        return bboxes[indx[:q_stop]], bboxes[indx[:b_stop]]  # query images, bboxes; query images <= bboxes
    
    def random_query(self, class_id: int, sigma: int=4.0) -> Image.Image:
        """
        Draws queries from the class, the amount of which follows
        a normal distribution with mean 1 and stddev 4
        """
        size = random.normalvariate(1.0, sigma)
        size = math.floor(abs(size)) + 1  # make sure that size is not 0
        folder_path = os.path.join(self.query_pool, str(class_id))
        queries = []
        indx = 0
        while indx < size:
            f = random.randint(0, self.query_instances[str(class_id)])
            f = folder_path + '/' + str(f) + '.jpg'
            if f not in queries:
                queries.append(f)
                indx += 1
        return [Image.open(i) for i in queries]

    def __len__(self):
        return len(self.keys)
    
    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()
        
        img_path = id_to_file(self.keys[index], self.image_dir)
        bboxes = self.image_ids[self.keys[index]]
        img_class: int = random.choices(list(bboxes.keys()), k=1)[0]  # randomly chose an image class to preform detection

        img = Image.open(img_path)  # the target image
        bboxes = np.asarray(bboxes[img_class])  # all the boxes in that class belonging to the target image
        
        # get queries
        if self.query_pool is not None and random.random() < self.case4_prob:
            # queries are pulled from disk at location self.query_pool
            queries = self.random_query(img_class, self.case4_sigma)
        else:
            # queries are pulled from the target image, and are a subset of bboxes
            queries, bboxes = self.random_query_bbox(bboxes)
            queries = [fetch_query(img, q) for q in queries]
        
        if self.query_transforms is not None:
            # pass empty dict as there are no boxes to be transformed
            queries = [self.query_transforms(q, {})[0] for q in queries]
        
        bboxes = torch.as_tensor(bboxes)
        # from [x, y, w, h] to [x, y, x1, y1], for transforms
        bboxes[:, 2] = bboxes[:, 0] + bboxes[:, 2]
        bboxes[:, 3] = bboxes[:, 1] + bboxes[:, 3]
        
        # format for transform functions
        bboxes = {'boxes': bboxes}

        if self.transforms is not None:
            img, bboxes = self.transforms(img, bboxes)
        
        # img.shape = [C, H, W]; both img, queries and bboxes are normalized
        return img, queries, bboxes['boxes']


def img_transforms(image_set):
    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]

    if image_set == 'train':
        return T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomSelect(
                T.RandomResize(scales, max_size=1333),
                T.Compose([
                    T.RandomResize([400, 500, 600]),
                    T.RandomSizeCrop(384, 600),
                    T.RandomResize(scales, max_size=1333),
                ])
            ),
            normalize,
        ])

    if image_set == 'val':
        return T.Compose([
            T.RandomResize([800], max_size=1333),
            normalize,
        ])

    raise ValueError(f'unknown {image_set}')

def query_transforms():
    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]

    return T.Compose([
        T.RandomHorizontalFlip(),
        T.RandomSelect(
            T.RandomResize(scales, max_size=1333),
            T.Compose([
                T.RandomResize([400, 500, 600]),
                T.RandomResize(scales, max_size=1333),
            ])
        ),
        normalize,
    ])

