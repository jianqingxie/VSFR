import json
import os
import h5py
import argparse

import numpy
import torch
import torch.nn as nn
from h5py._hl import base
from tqdm import tqdm


class DataProcessor(nn.Module):
    def __init__(self):
        super(DataProcessor, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d((7, 7))

    def forward(self, x):
        x = self.pool(x)
        x = torch.squeeze(x)    # [1, d, h, w] => [d, h, w] 
        x = x.permute(1, 2, 0)  # [d, h, w] => [h, w, d]
        return x.view(-1, x.size(-1))   # [h*w, d]


def save_dataset(file_path, feat_paths): 
    print('save the ori grid features to the features with specified size')
    processor = DataProcessor()  
    with h5py.File(file_path, 'w') as f:
        for i in tqdm(range(len(feat_paths))):
            feat_path = os.path.join('/mnt/data/X101-features', 'train2014', feat_paths[i])
            if not os.path.exists(feat_path):
                feat_path = os.path.join('/mnt/data/X101-features', 'val2014', feat_paths[i])

            img_name = feat_path.split('/')[-1]

            img_feat = torch.load(feat_path)   
            img_id = int(img_name.split('.')[0])

            img_feat = processor(img_feat)
            data = numpy.asarray(img_feat.numpy(), order="C", dtype=base.guess_dtype(img_feat.numpy()))
            f.require_dataset('%d_grids' % img_id, shape=data.shape, dtype=data.dtype, data=img_feat.numpy())
        f.close()


def get_feat_paths(dir_to_save_feats, data_split='trainval', test2014_info_path=None):
    print('get the paths of grid features')
    ans = []
    if data_split == 'trainval':
        ans = os.listdir(os.path.join(dir_to_save_feats, 'train2014')) + os.listdir(os.path.join(dir_to_save_feats, 'val2014'))
    elif data_split == 'test':
        assert test2014_info_path is not None
        with open(test2014_info_path, 'r') as f:
            test2014_info = json.load(f)

        for image in test2014_info['images']:
            img_id = image['id']
            feat_path = os.path.join(dir_to_save_feats, 'test2015', img_id+'.pth')
            assert os.path.exists(feat_path)
            ans.append(feat_path)
        assert len(ans) == 40775
    assert ans      # make sure ans list is not empty
    return ans
    

def main(args):
    feat_paths = get_feat_paths(args.dir_to_save_feats, args.data_split, args.test2014_info_path)
    file_path = os.path.join(args.dir_to_save_feats, 'X101_grid_feats_coco_'+args.data_split+'.hdf5')
    save_dataset(file_path, feat_paths)
    print('finished!')


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='data process') 
    parser.add_argument('--dir_to_save_feats', type=str, default='/mnt/data/X101-features')
    parser.add_argument('--data_split', type=str, default='trainval')   # trainval, test
    parser.add_argument('--test2014_info_path', type=str, default=None)      # None, image_info_test2014.json
    args = parser.parse_args()
        
    main(args)

