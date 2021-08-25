from collections import OrderedDict
from utils.ops import nested_tensor_from_tensor_list
from data.dataset import *

class FeatureQueries(Dataset):
    """For the feature extractor"""
    def __init__(self, query_pool):
        with open(os.path.join(query_pool, 'instances.json'), 'r') as f:
            # query instances describes range of file names under each class
            # to prevent loading all file paths into memory
            self.query_instances = json.load(f)
        self.query_pool = query_pool
        self.classes = [k for k in self.query_instances]
        self.classes.sort()
        self.len = sum([self.query_instances[k] for k in self.classes])
        self.transforms = query_transforms()
        print('loaded instances')
    
    def __len__(self):
        return self.len
    
    def __getitem__(self, index):
        collected_sum = 0
        for k in self.classes:
            collected_sum += self.query_instances[k]
            if collected_sum > index:
                im_path = os.path.join(self.query_pool, k, str(collected_sum - index - 1) + '.jpg')
                im = Image.open(im_path).convert('RGB')
                im, imt = self.transforms(im, {})[0], self.transforms(im, {})[0]
                return im, imt, k
    
    @staticmethod
    def collate_fn(batch):
        im_o = OrderedDict()
        im_t = OrderedDict()
        for im, im_, c in batch:
            if c not in im_o:
                im_o[c] = [im]
                im_t[c] = [im_]
            else:
                im_o[c].append(im)
                im_t[c].append(im_)
        if random.uniform(0, 1) > 0.5:
            for k in im_t:
                random.shuffle(im_t[k])
        im_o = nested_tensor_from_tensor_list([x for k in im_o for x in im_o[k]], exclude_mask_dim=-2)
        im_t = nested_tensor_from_tensor_list([x for k in im_t for x in im_t[k]], exclude_mask_dim=-2)
        return im_o[0], im_t[0]

