# %%
import os
import json
import random
from typing import List

import torch
import torchvision
import torchvision.transforms.functional as F
from PIL import Image
import numpy as np

import data.transforms as T


class CocoDetection(torchvision.datasets.CocoDetection):
    def __init__(self, img_folder, ann_file, transforms, get_query):
        super(CocoDetection, self).__init__(img_folder, ann_file)
        self._transforms = transforms
        self.prepare = VectorizeData()
        self.get_query = get_query
        self.normalize = T.Compose([T.ToTensor(),
                               T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    def __getitem__(self, idx):
        """target['boxes'] = [x, y, x1, y1] top left corner, bottom right corner"""
        img, target = super(CocoDetection, self).__getitem__(idx)
        image_id = self.ids[idx]
        target = {'image_id': image_id, 'annotations': target}
        img, target = self.prepare(img, target)

        if self._transforms is not None:
            img, target = self._transforms(img, target)
        img, qrs, target = self.get_query(img, target)

        img, target = self.normalize(img, target)
        qrs = [self.normalize(qr, {})[0] for qr in qrs]
        return img, qrs, target


class VectorizeData:
    """Tensorizes data output"""
    def __call__(self, image, target):
        w, h = image.size

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

        if keypoints is not None:
            keypoints = keypoints[keep]

        target = {}
        target["boxes"] = boxes
        target["labels"] = classes
        if keypoints is not None:
            target["keypoints"] = keypoints

        # for conversion to coco api
        iscrowd = torch.tensor([obj["iscrowd"] if "iscrowd" in obj else 0 for obj in anno])
        target["iscrowd"] = iscrowd[keep]

        target["orig_size"] = torch.as_tensor([int(h), int(w)])
        target["size"] = torch.as_tensor([int(h), int(w)])

        return image, target


class GetQuery:
    def __init__(self, transforms, mean_amount, std_amount, min_size=None, stretch_limit=0.3, query_pool=None, max_queries=10, prob_pool=0.1) -> None:
        self.transforms = transforms
        self.mean = mean_amount
        self.std = std_amount
        self.min_size = min_size
        self.stretch_limit = stretch_limit
        self.max_queries = max_queries
        self.query_pool = query_pool
        self.prob_pool = max(0, min(1, prob_pool))

        if query_pool is not None:
            with open(os.path.join(query_pool, 'instances.json'), 'r') as f:
                # query instances describes range of file names under each class
                # to prevent loading all file paths into memory
                self.query_instances = json.load(f)

    def random_size(self) -> int:
        return min(int(abs(random.normalvariate(self.mean - 1, self.std)) + 1), self.max_queries)

    def fetch_query(self, im, boxes) -> List[Image.Image]:
        boxes = boxes.tolist()
        return [F.crop(im, b[1], b[0], b[3], b[2]) for b in boxes]
    
    def stretch_queries(self, queries: List[Image.Image]) -> List[Image.Image]:
        """resizes queries to make them more similar in shape"""
        stretch = random.uniform(0, self.stretch_limit)
        widths = np.array([im.width for im in queries])
        heights = np.array([im.height for im in queries])
        w_dist = (widths - widths.mean()) * stretch
        h_dist = (heights - heights.mean()) * stretch
        widths = widths - w_dist
        heights = heights - h_dist
        queries = [im.resize((int(w), int(h))) for im, w, h in zip(queries, widths, heights)]
        return queries

    def random_class(self, target):
        classes = target['labels']
        if classes.shape[0] == 0:
            return
        keep = torch.multinomial(torch.ones_like(classes, dtype=torch.float32), num_samples=1)
        keep = classes[keep] == classes
        target['boxes'] = target['boxes'][keep]
        target['obj label'] = int(classes[keep][0])
    
    def sample_query_from_pool(self, samples, label):
        folder_path = os.path.join(self.query_pool, str(label))
        queries = []
        indx = 0
        while indx < samples:
            f = random.randint(0, self.query_instances[str(label)] - 1)
            f = folder_path + '/' + str(f) + '.jpg'
            im = Image.open(f)
            if self.min_size is None:
                queries.append(im.convert('RGB'))
                indx += 1
            elif im.width > self.min_size and im.height > self.min_size:
                queries.append(im.convert('RGB'))
                indx += 1
        return queries
        
    def random_query(self, img, target) -> List[Image.Image]:
        bboxes = target['boxes'].clone()
        # dealing with zero box cases
        if bboxes.shape[0] == 0 and self.query_pool is not None:
            # get a random sample from the query pool
            label = random.sample(self.query_instances.keys(), 1)[0]
            return self.sample_query_from_pool(self.random_size(), label)
        elif bboxes.shape[0] == 0:
            # get some random noise
            sizes = [(random.randint(32, 128), random.randint(32, 128)) for _ in range(self.random_size())]
            return [Image.fromarray(np.random.randint(0, 255, size=(h, w, 3)), mode='RGB') for (h, w) in sizes]

        bboxes[:, 2:] -= bboxes[:, :2]
        if self.min_size is not None:
            keep = (bboxes[:, 2] > self.min_size).logical_and(bboxes[:, 3] > self.min_size)
            # size could be zero, filter by or instead
            if not keep.any():
                keep = (bboxes[:, 2] > self.min_size).logical_or(bboxes[:, 3] > self.min_size)
            if keep.any():
                bboxes = bboxes[keep]
        
        areas = bboxes[:, 2] * bboxes[:, 3]
        areas = areas / areas.max()
        samples = min(bboxes.shape[0], self.random_size())
        if self.query_pool is not None:
            samples_pool = int(samples * self.prob_pool + 0.5)
            samples_im = samples - samples_pool
            query_im = self.sample_query_from_pool(samples_pool, target['obj label'])
        else:
            samples_im = samples
            query_im = []
        if samples_im > 0:
            keep = torch.multinomial(areas, samples_im, replacement=False)
            query = bboxes[keep]
            query_im.extend(self.fetch_query(img, query))
        return query_im
    
    def __call__(self, img, target):
        self.random_class(target)
        qrs = self.random_query(img, target)
        qrs = [self.transforms(qr, {})[0] for qr in qrs]
        qrs = self.stretch_queries(qrs)
        # every instance could be of the 'object' category
        target['labels'] = torch.zeros(target['boxes'].shape[0], dtype=torch.int64)
        return img, qrs, target


def img_transforms(image_set):
    scales = [352, 384, 416, 448, 480, 512, 544, 576, 608, 640, 672]

    if image_set == 'train':
        return T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomSelect(
                T.RandomResize(scales, max_size=672),
                T.Compose([
                    T.RandomResize([400, 500, 600]),
                    #T.RandomSizeCrop(384, 600),
                    T.RandomResize(scales, max_size=672),
                ])
            ),
        ])

    if image_set == 'val':
        return T.Compose([
            T.RandomResize([512], max_size=672),
        ])

    raise ValueError(f'unknown {image_set}')


def query_transforms():
    scales = [96, 128, 160, 192, 224, 256]

    return T.Compose([
        T.RandomHorizontalFlip(),
        T.RandomSelect(
            T.RandomResize(scales, max_size=256),
            T.Compose([
                T.RandomResize([64, 100, 200], max_size=256),
                T.RandomResize(scales, max_size=256),
            ])
        ),
        #T.CompleteAugment(),
    ])


def collate_fn(batch):
    img = [i[0] for i in batch]
    query_im = [i[1] for i in batch]
    #similarity_scores = torch.tensor([[i[3]] for i in batch])
    target = [i[2] for i in batch]
    return img, query_im, target


# TODO decrease similarity as a function of steps/loss? 
if __name__ == "__main__":
    from utils.misc import visualize_output
    from torch.utils.data import DataLoader
    from tqdm import tqdm
    root = "datasets/coco/"
    proc = 'train'
    query_process = GetQuery(query_transforms(), 3, 2, stretch_limit=0.8, query_pool=root + proc + "2017_query_pool")

    dataset = CocoDetection(root + proc + '2017', root + f'annotations/instances_{proc}2017.json', img_transforms('train'), query_process)

    #val_set = DataLoader(dataset, 16, True, num_workers=12, collate_fn=collate_fn)
    #val_set = iter(val_set)
    #
    #for im, qrs, tgt in tqdm(val_set):
    #    pass

    dataset.get_query.prob_pool = 0.3
    dataset.get_query.min_size = 64
    im, qrs, tgt = dataset[92]
    visualize_output(im, qrs, tgt['boxes'])
