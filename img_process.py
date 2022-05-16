import os
import shutil
import random
from tqdm import tqdm
from PIL import Image, ImageEnhance


def transpose_LEFT_RIGHT(image):  # 水平翻转
    return image.transpose(Image.FLIP_LEFT_RIGHT)


def random_brightness(image):  # 亮度
    brightness = random.uniform(0.5, 1.5)
    enh_bri = ImageEnhance.Brightness(image)
    img_brightened = enh_bri.enhance(brightness)
    return img_brightened


def random_contrast(image):  # 对比度
    contrast = random.uniform(0.5, 1.5)
    enh_con = ImageEnhance.Contrast(image)
    img_contrasted = enh_con.enhance(contrast)
    return img_contrasted


def random_color(image):  # 色度
    color = random.uniform(0.5, 1.5)
    enh_col = ImageEnhance.Contrast(image)
    img_colored = enh_col.enhance(color)
    return img_colored


def random_sharpness(image):  # 锐度
    sharpness = random.uniform(0.5, 1.5)
    enh_sha = ImageEnhance.Contrast(image)
    img_sharped = enh_sha.enhance(sharpness)
    return img_sharped


# 扩充数据集，先通过翻转将所有类别的数据集扩充到相同数量，再随机增强亮度、对比度、色度、锐度
def random_expand(content_ori, content_tra, content_tra_enh, num_after_trans=600):
    # 通过翻转将不同类别的数据集扩充到相同数量
    for cls in os.listdir(content_ori):
        if cls == 'test':  # 测试集不做处理
            continue
        content_ori_cls = os.path.join(content_ori, cls)  # 原图片所在目录
        content_tra_cls = os.path.join(content_tra, cls)  # 翻转后图片保存目录

        # 清空目录内容
        if os.path.exists(content_tra_cls):
            shutil.rmtree(content_tra_cls)
        os.mkdir(content_tra_cls)

        # 翻转
        trans_num = num_after_trans - len(os.listdir(content_ori_cls))  # 待翻转的图片数量
        trans_set = random.sample(os.listdir(content_ori_cls), trans_num)  # 随机选出trans_num数量的图片用于翻转
        print('正在进行随机翻转...')
        for path_img in tqdm(os.listdir(content_ori_cls)):
            img = Image.open(content_ori_cls + path_img)
            shutil.copy(content_ori_cls + path_img, content_tra_cls + path_img)
            if path_img in trans_set:
                transpose_LEFT_RIGHT(img).save(content_tra_cls + path_img[:-4] + '_tra.png')

    # 随机增强亮度、对比度、色度、锐度，进一步扩充数据集
    for cls in os.listdir(content_tra):
        if cls == 'test':
            continue
        content_tra_cls = content_tra + cls + '/'  # 翻转后图片保存目录
        content_enh_cls = content_tra_enh + cls + '/'  # 增强后图片保存目录

        # 清空目录内容
        if os.path.exists(content_enh_cls):
            shutil.rmtree(content_enh_cls)
        os.makedirs(content_enh_cls)

        print('正在进行随机增强...')
        for path_img in tqdm(os.listdir(content_tra_cls)):
            img = Image.open(content_tra_cls + path_img)
            shutil.copy(content_tra_cls + path_img, content_enh_cls + path_img)
            random_brightness(img).save(content_enh_cls + path_img[:-4] + '_bri.png')
            random_contrast(img).save(content_enh_cls + path_img[:-4] + '_con.png')
            random_color(img).save(content_enh_cls + path_img[:-4] + '_col.png')
            random_sharpness(img).save(content_enh_cls + path_img[:-4] + '_sha.png')


if __name__ == '__main__':
    content_ori = 'data_ori/datasets/train/'
    content_tra = 'data_ori/datasets/train_tra/'
    content_tra_enh = 'data/train/'
    for content in [content_tra, content_tra_enh]:
        if not os.path.exists(content):
            os.mkdir(content)
    random_expand(content_ori, content_tra, content_tra_enh)