import os
import sys
from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

num_classes = 4
img_size = 224

# 模型保存文件夹
path_weights = ROOT / '../drive/MyDrive/weights_convnext_multi'

# 保存模型路径 & predict时调用的模型
# model = path_weights / 'best_model.pth'
# model = ROOT / 'weights' / 'best_model_l1k_epoch6_10fold.pth'
# model = ROOT / 'weights' / 'best_model_l1k_94.pth'


# 数据集&分类标签 路径
# path_train = ROOT / 'data/train/'
# path_test = ROOT / 'data/test'
path_train = ROOT / 'data_add_extra/train/'
path_test = ROOT / 'data_add_extra/test'
path_test_submit = '../../test'
path_json = ROOT / 'class_indices.json'