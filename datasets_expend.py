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
    sharpness = random.uniform(0.5, 2)
    enh_sha = ImageEnhance.Contrast(image)
    img_sharped = enh_sha.enhance(sharpness)
    return img_sharped


# 扩充数据集，先通过翻转将所有类别的数据集扩充到相同数量，再随机增强亮度、对比度、色度、锐度
def random_expand(content_ori, content_flip, content_flip_enh):
    num_cls = []  # 各类的训练集数目
    for cls in os.listdir(content_ori):
        num_cls.append(len(os.listdir(os.path.join(content_ori, cls))))
    print('The number of images in each category: ', num_cls)

    # 设定翻转后及增强后 扩充的 每类的图片数目
    # 若原始图片数量<num_after_flip,则翻转扩充;否则不翻转
    # 例如,num_cls = [415,123,187,177],则扩充到2 * int(min(415,123,187,177) / 100) * 100 = 2*100
    num_after_flip = 2 * int(min(num_cls) / 100) * 100
    # 随机改变亮度、对比度、色度、锐度，扩充后就是5倍
    num_after_enh = 5 * num_after_flip

    # 通过翻转将不同类别的数据集扩充到相同数量
    for index, cls in enumerate(os.listdir(content_ori)):
        content_ori_cls = os.path.join(content_ori, cls)  # 原图片所在目录
        content_flip_cls = os.path.join(content_flip, cls)  # 翻转后图片保存目录

        # 删除目录及所有内容
        if os.path.exists(content_flip_cls):
            shutil.rmtree(content_flip_cls)

        print('Copying in flip...')
        shutil.copytree(content_ori_cls, content_flip_cls)

        if num_cls[index] < num_after_flip:    # 该类训练集数目不足，翻转扩充
            num_flip = num_after_flip - num_cls[index]  # 待翻转的图片数量
            set_flip = random.sample(os.listdir(content_ori_cls), num_flip)  # 随机选出trans_num数量的图片用于翻转
            print(f'The dataset of class {cls} is being flipped...')
            for path_img in tqdm(os.listdir(content_ori_cls)):
                img = Image.open(os.path.join(content_ori_cls, path_img))
                if path_img in set_flip:
                    img_name = path_img[:-4] + '_fli.png'
                    transpose_LEFT_RIGHT(img).save(os.path.join(content_flip_cls, path_img[:-4] + '_fli.png'))
        elif num_cls[index] % 4: # 该类训练集数目充足，但不是4的倍数（即后面增强扩展后无法达到相同数目，如415无法扩展到1000）
            # 扩充到4的倍数
            num_flip = 4 - num_cls[index] % 4  # 待翻转的图片数量
            set_flip = random.sample(os.listdir(content_ori_cls), num_flip)  # 随机选出num_flip数量的图片用于翻转
            print(f'The dataset of class {cls} is being flipped...')
            for path_img in tqdm(os.listdir(content_ori_cls)):
                img = Image.open(os.path.join(content_ori_cls, path_img))
                if path_img in set_flip:
                    img_name = path_img[:-4] + '_fli.png'
                    transpose_LEFT_RIGHT(img).save(os.path.join(content_flip_cls, path_img[:-4] + '_fli.png'))

    # 随机增强亮度、对比度、色度、锐度，进一步扩充数据集
    for cls in os.listdir(content_flip):
        content_flip_cls = os.path.join(content_flip, cls)  # 翻转后图片保存目录
        content_flip_enh_cls = os.path.join(content_flip_enh, cls)  # 增强后图片保存目录

        # 删除目录及所有内容
        if os.path.exists(content_flip_enh_cls):
            shutil.rmtree(content_flip_enh_cls)

        print('Copying in enhance...')
        shutil.copytree(content_flip_cls, content_flip_enh_cls)

        print(f'The dataset of class {cls} is being enhanced...')
        if len(os.listdir(content_flip_cls)) > num_after_flip:
            num_enh = int((num_after_enh - len(os.listdir(content_flip_cls))) / 4)
            set_enh = random.sample(os.listdir(content_flip_cls), num_enh)  # 随机选出num_enh数量的图片用于增强
            for path_img in tqdm(os.listdir(content_flip_cls)):
                if path_img in set_enh:
                    img = Image.open(os.path.join(content_flip_cls, path_img))
                    random_brightness(img).save(os.path.join(content_flip_enh_cls, path_img[:-4] + '_bri.png'))
                    random_contrast(img).save(os.path.join(content_flip_enh_cls, path_img[:-4] + '_con.png'))
                    random_color(img).save(os.path.join(content_flip_enh_cls, path_img[:-4] + '_col.png'))
                    random_sharpness(img).save(os.path.join(content_flip_enh_cls, path_img[:-4] + '_sha.png'))
        else:
            for path_img in tqdm(os.listdir(content_flip_cls)):
                img = Image.open(os.path.join(content_flip_cls, path_img))
                random_brightness(img).save(os.path.join(content_flip_enh_cls, path_img[:-4] + '_bri.png'))
                random_contrast(img).save(os.path.join(content_flip_enh_cls, path_img[:-4] + '_con.png'))
                random_color(img).save(os.path.join(content_flip_enh_cls, path_img[:-4] + '_col.png'))
                random_sharpness(img).save(os.path.join(content_flip_enh_cls, path_img[:-4] + '_sha.png'))
        break


if __name__ == '__main__':
    # 训练集
    content_ori = 'data_ori/train'  # 原始训练集目录
    content_flip = 'data_ori/train_flip'  # 翻转扩充后的训练集目录
    content_flip_enh = 'data/train'  # 增强扩充后的训练集目录
    for content in [content_flip, content_flip_enh]:
        if not os.path.exists(content):
            os.makedirs(content)
    # 测试集
    content_test_ori = 'data_ori/test'  # 原始测试集目录
    content_test_copy = 'data/test'  # 复制到的测试集目录
    if os.path.exists(content_test_copy):
        shutil.rmtree(content_test_copy)
    shutil.copytree(content_test_ori, content_test_copy)  # 测试集从data_ori复制到data目录下

    # 最终效果: 训练与测试时的数据集都使用data目录下的文件
    random_expand(content_ori, content_flip, content_flip_enh)