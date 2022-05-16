import os
import json
import argparse

import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np

from model import convnext_large as create_model  # 更改预训练模型时更改import内容
import params


def main(args):
    print(args)
    device = args.device
    print(f"using {device} device.")

    num_classes = args.num_classes
    img_size = 224
    data_transform = transforms.Compose(
        [transforms.Resize(int(img_size * 1.14)),
         transforms.CenterCrop(img_size),
         transforms.ToTensor(),
         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    # read class_indict
    json_path = args.path_json
    assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)

    with open(json_path, "r") as f:
        class_indict = json.load(f)

    # create model
    model = create_model(num_classes=num_classes).to(device)
    # load model weights
    model_weight_path = args.weights
    model.load_state_dict(torch.load(model_weight_path, map_location=device))
    model.eval()

    test_path = args.path_test
    TP, FN, FP, TN = 0, 0, 0, 0

    img_name_list = os.listdir(test_path)
    # 预测结果列表
    ans = []
    for img_name in img_name_list:
        # 拿到图片编号
        img_num = img_name.split('.')[0]
        img_path = os.path.join(test_path, img_name)
        img = Image.open(img_path)
        # [N, C, H, W]
        img = data_transform(img)
        # expand batch dimension
        img = torch.unsqueeze(img, dim=0)

        with torch.no_grad():
            output = torch.squeeze(model(img.to(device))).cpu()
            predict = torch.softmax(output, dim=0)
            predict_cls = torch.argmax(predict).numpy()

            # print("class: {}   prob: {:.3}".format(class_indict[str(predict_cls)], predict[predict_cls].numpy()))

        ans.append([str(img_name), int(predict_cls)])
    # 保存结果，不需要其他相关的分隔符，保存类型等参数
    np.save("./pre.npy", ans)



def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=params.model, help='model path(s)')  # 模型参数
    parser.add_argument('--path_test', type=str, default=params.path_test_submit, help='test datasets path')  # 测试集路径
    parser.add_argument('--path_json', type=str, default=params.path_json, help='class_indice.json path')
    parser.add_argument('--num_classes', type=int, default=params.num_classes, help='number of classes')
    parser.add_argument('--device', default=params.device, help='device id (i.e. 0 or 0,1 or cpu)')

    opt = parser.parse_args()
    return opt


if __name__ == '__main__':
    opt = parse_opt()
    main(opt)