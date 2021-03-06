# Diabetic-Retinopathy-Detection-multi_class-ConvNeXt
use ConvNeXt for multi classification  
task: Diabetic-Retinopathy-Detection

数据划分：
1. 按类别9:1划分训练集（这里不分训练集和验证集，都归为训练集）和测试集 
2. 训练时8:2划分训练集和验证集

data：  
数据扩展（训练集）：  
假设各类训练集中图片数量 num_cls = [num1, num2, num3, ...]  
目标：将每类的训练集都扩充到相同数目  
步骤：
1. 设置 num_after_flip = 2 * int(min(num_cls) / 100) * 100
2. 设置 num_after_enh = 5 * num_after_flip，下面会讲原因
3. 先通过翻转扩充训练集（这里只有水平翻转，因为本数据集垂直翻转变化不大），即将每类数据集扩充到 num_after_flip 张。若第k类训练集中图片数量numk > num_after_flip, 则不做处理（也不算完全不处理）。
4. 然后选择合适数量的图片（包括翻转后的）进行随机增强，对于选出的每张图片进行随机增强 亮度、对比度、色度、锐度（每张图片四种增强都进行处理并分别保存），这样扩充后就是原来的5倍了，这就是这么设置'步骤2'公式的原因
5. 至此，每类的训练集都扩充到相同数目

例子（本数据集）：  
num_cls = [415, 123, 187, 177]  
步骤：  
1. num_after_flip = 2 * int(min(num_cls) / 100) * 100 = 2 * int(123/100) * 100 = 200
2. num_after_enh = 5 * num_after_flip = 5 * 200 = 1000
3. 第2、3、4类，每类数量　< num_after_flip，因此分别随机翻转200-123，200-187，200-177张；第1类 415张 > num_after_flip，不做翻转，但为了让第1类训练集扩充到1000张，因此待增强的图片需要是4的倍数，有数学原理易知原始训练集需要是4的倍数（假设a是4的倍数，1000-a也是4的倍数），所以第1类也需要随机翻转1张图片，变为416张（是4的倍数）
4. 第2、3、4类，每张图片都进行随机增强，每类都由200张扩充到1000张；第1类，随机翻转后 num1 = 416 > num_after_flip，随机取出 (num_after_enh - num1)/4 = (1000-416)/4 = 146张进行增强，这样第1类最终有 416 + 146 * 4 = 1000张
5. 至此，每类的训练集都扩充到相同数目（1000张）  

data_add_extra:  
在Kaggle竞赛官网[Diabetic Retinopathy Detection](https://www.kaggle.com/competitions/diabetic-retinopathy-detection/data)中扩充了一些  
