from itertools import product
import random

from random import choice
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import (CenterCrop, Compose, InterpolationMode,
                                    Normalize, RandomHorizontalFlip,
                                    RandomPerspective, RandomRotation, Resize,
                                    ToTensor)
from torchvision.transforms.transforms import RandomResizedCrop

BICUBIC = InterpolationMode.BICUBIC
n_px = 224


def transform_image(split="train", imagenet=False):
    if imagenet:
        # from czsl repo.
        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        transform = Compose(
            [
                RandomResizedCrop(n_px),
                RandomHorizontalFlip(),
                ToTensor(),
                Normalize(
                    mean,
                    std,
                ),
            ]
        )
        return transform

    if split == "test" or split == "val":
        transform = Compose(
            [
                Resize(n_px, interpolation=BICUBIC),
                CenterCrop(n_px),
                lambda image: image.convert("RGB"),
                ToTensor(),
                Normalize(
                    (0.48145466, 0.4578275, 0.40821073),
                    (0.26862954, 0.26130258, 0.27577711),
                ),
            ]
        )
    else:
        transform = Compose(
            [
                # RandomResizedCrop(n_px, interpolation=BICUBIC),
                Resize(n_px, interpolation=BICUBIC),
                CenterCrop(n_px),
                RandomHorizontalFlip(),
                RandomPerspective(),
                RandomRotation(degrees=5),
                lambda image: image.convert("RGB"),
                ToTensor(),
                Normalize(
                    (0.48145466, 0.4578275, 0.40821073),
                    (0.26862954, 0.26130258, 0.27577711),
                ),
            ]
        )

    return transform

class ImageLoader:
    def __init__(self, root):
        self.img_dir = root

    def __call__(self, img):
        file = '%s/%s' % (self.img_dir, img)
        img = Image.open(file).convert('RGB')
        return img


class CompositionDataset(Dataset):
    def __init__(
            self,
            root,
            phase,
            split='compositional-split-natural',
            open_world=False,
            imagenet=False,
            same_prim_sample=False
    ):
        self.root = root
        self.phase = phase
        self.split = split
        self.open_world = open_world
        self.same_prim_sample = same_prim_sample

        self.feat_dim = None
        self.transform = transform_image(phase, imagenet=imagenet)
        self.loader = ImageLoader(self.root + '/images/')

        self.attrs, self.objs, self.pairs, \
                self.train_pairs, self.val_pairs, \
                self.test_pairs = self.parse_split()

        if self.open_world:
            self.pairs = list(product(self.attrs, self.objs))

        self.train_data, self.val_data, self.test_data = self.get_split_info()
        if self.phase == 'train':
            self.data = self.train_data
        elif self.phase == 'val':
            self.data = self.val_data
        else:
            self.data = self.test_data

        self.obj2idx = {obj: idx for idx, obj in enumerate(self.objs)}
        self.attr2idx = {attr: idx for idx, attr in enumerate(self.attrs)}
        self.pair2idx = {pair: idx for idx, pair in enumerate(self.pairs)}

        print('# train pairs: %d | # val pairs: %d | # test pairs: %d' % (len(
            self.train_pairs), len(self.val_pairs), len(self.test_pairs)))
        print('# train images: %d | # val images: %d | # test images: %d' %
              (len(self.train_data), len(self.val_data), len(self.test_data)))

        self.train_pair_to_idx = dict(
            [(pair, idx) for idx, pair in enumerate(self.train_pairs)]
        )

        if self.open_world:
            mask = [1 if pair in set(self.train_pairs) else 0 for pair in self.pairs]
            self.seen_mask = torch.BoolTensor(mask) * 1.

            self.obj_by_attrs_train = {k: [] for k in self.attrs}
            for (a, o) in self.train_pairs:
                self.obj_by_attrs_train[a].append(o)

            # Intantiate attribut-object relations, needed just to evaluate mined pairs
            self.attrs_by_obj_train = {k: [] for k in self.objs}
            for (a, o) in self.train_pairs:
                self.attrs_by_obj_train[o].append(a)

        if self.phase == 'train' and self.same_prim_sample:
            self.same_attr_diff_obj_dict = {pair: list() for pair in self.train_pairs}
            self.same_obj_diff_attr_dict = {pair: list() for pair in self.train_pairs}
            self.diff_attr_diff_obj_dict = {pair: list() for pair in self.train_pairs}
            for i_sample, sample in enumerate(self.train_data):
                sample_attr, sample_obj = sample[1], sample[2]
                for pair_key in self.same_attr_diff_obj_dict.keys():
                    if (pair_key[1] == sample_obj) and (pair_key[0] != sample_attr):
                        self.same_obj_diff_attr_dict[pair_key].append(i_sample)
                    elif (pair_key[1] != sample_obj) and (pair_key[0] == sample_attr):
                        self.same_attr_diff_obj_dict[pair_key].append(i_sample)
                    elif (pair_key[1] != sample_obj) and (pair_key[0] != sample_attr):
                        self.diff_attr_diff_obj_dict[pair_key].append(i_sample)

        # if self.phase == 'train' and self.same_prim_sample:
        #     self.attr_num = {attr: 0 for idx, attr in enumerate(self.attrs)}
        #     self.obj_num = {obj: 0 for idx, obj in enumerate(self.objs)}
        #     for i_sample, sample in enumerate(self.train_data):
        #         sample_attr, sample_obj = sample[1], sample[2]
        #         for pair_key in self.attr_num.keys():
        #             if pair_key == sample_attr:
        #                 self.attr_num[pair_key] = self.attr_num[pair_key] + 1
                
        #         for pair_key in self.obj_num.keys():
        #             if pair_key == sample_obj:
        #                 self.obj_num[pair_key] = self.obj_num[pair_key] + 1
        #     self.attr_num1 = {attr: self.attr_num[attr]/22998*100 for idx,attr in enumerate(self.attr_num)}
        #     self.obj_num1 = {obj: self.obj_num[obj]/22998*100 for idx,obj in enumerate(self.obj_num)}
            
        #     print("attr",self.attr_num1)
        #     print("obj",self.obj_num1)
                
                 



    def get_split_info(self):
        data = torch.load(self.root + '/metadata_{}.t7'.format(self.split))
        train_data, val_data, test_data = [], [], []
        for instance in data:
            image, attr, obj, settype = instance['image'], instance[
                'attr'], instance['obj'], instance['set']

            if attr == 'NA' or (attr,
                                obj) not in self.pairs or settype == 'NA':
                continue

            data_i = [image, attr, obj]
            if settype == 'train':
                train_data.append(data_i)
            elif settype == 'val':
                val_data.append(data_i)
            else:
                test_data.append(data_i)

        return train_data, val_data, test_data

    def parse_split(self):
        def parse_pairs(pair_list):
            with open(pair_list, 'r') as f:
                pairs = f.read().strip().split('\n')
                # pairs = [t.split() if not '_' in t else t.split('_') for t in pairs]
                pairs = [t.split() for t in pairs]
                pairs = list(map(tuple, pairs))
            attrs, objs = zip(*pairs)
            return attrs, objs, pairs

        tr_attrs, tr_objs, tr_pairs = parse_pairs(
            '%s/%s/train_pairs.txt' % (self.root, self.split))
        vl_attrs, vl_objs, vl_pairs = parse_pairs(
            '%s/%s/val_pairs.txt' % (self.root, self.split))
        ts_attrs, ts_objs, ts_pairs = parse_pairs(
            '%s/%s/test_pairs.txt' % (self.root, self.split))

        all_attrs, all_objs = sorted(
            list(set(tr_attrs + vl_attrs + ts_attrs))), sorted(
                list(set(tr_objs + vl_objs + ts_objs)))
        all_pairs = sorted(list(set(tr_pairs + vl_pairs + ts_pairs)))

        return all_attrs, all_objs, all_pairs, tr_pairs, vl_pairs, ts_pairs

    def __getitem__(self, index):
        image, attr, obj = self.data[index]
        img = self.loader(image)
        img = self.transform(img)

        if self.phase == 'train':
            data = [
                img, self.attr2idx[attr], self.obj2idx[obj], self.train_pair_to_idx[(attr, obj)]
            ]
        else:
            data = [
                img, self.attr2idx[attr], self.obj2idx[obj], self.pair2idx[(attr, obj)]
            ]

        if self.phase == 'train' and self.same_prim_sample:
            # [same_attr_image, same_attr, diff_obj], same_attr_mask = self.same_A_diff_B(label_A=attr, label_B=obj, phase='attr')
            # [same_obj_image, diff_attr, same_obj], same_obj_mask = self.same_A_diff_B(label_A=obj, label_B=attr, phase='obj')
            same_attr_image, same_attr_mask = self.same_A_diff_B(label_A=attr, label_B=obj, phase='attr',index=index)
            same_obj_image, same_obj_mask = self.same_A_diff_B(label_A=obj, label_B=attr, phase='obj',index=index)
            diff_image, diff_obj_mask = self.diff_A_diff_B(label_A=attr, label_B=obj)
            for i in range(1):
                same_attr_image[i] = self.transform(self.loader(same_attr_image[i]))
                same_obj_image[i] = self.transform(self.loader(same_obj_image[i]))
            for i in range(1):
                diff_image[i] = self.transform(self.loader(diff_image[i]))
            data += [same_attr_image,same_obj_image,diff_image]

        return data

    def same_A_diff_B(self, label_A, label_B, phase='attr',index=1):
        if phase=='attr':
            candidate_list = self.same_attr_diff_obj_dict[(label_A, label_B)]
        else:
            candidate_list = self.same_obj_diff_attr_dict[(label_B,  label_A)]
        if len(candidate_list) != 0:
            idx = random.sample(candidate_list, 1)
            mask = 1
        else:
            idx = [index]
            mask = 0
        retdata = []
        for i in idx:
            retdata.append(self.data[i][0])
        return retdata, mask
    
    def diff_A_diff_B(self, label_A, label_B, phase='attr'):
        
        candidate_list = self.diff_attr_diff_obj_dict[(label_A, label_B)]
        
        if len(candidate_list) != 0:
            idx = random.sample(candidate_list, 10)
            mask = 1
        retdata = []
        for i in idx:
            retdata.append(self.data[i][0])
        return retdata, mask

    def __len__(self):
        return len(self.data)
