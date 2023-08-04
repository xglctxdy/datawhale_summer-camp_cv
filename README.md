# datawhale_summer-camp_cv
datawhale夏令营cv方向
其中chapter_0是环境配置的一些事项  
另外png和自定义数据集是当时群里面其他人写的一些小tricks
## 笔记
记录了一些我遇到的问题和思考
## baseline
在这里列出了助教提供的两个版本，其中粗浅的做了一点小改动
1. 基于逻辑回归
2. 基于cnn
## 工具
编写了nii转换为jpg文件的脚本，即nii _to_jpg.ipynb
## 个人的优化和改进
为my_cnn实际效果并不好，目前位找到原因，不如微调原版baseline  
my_cnn为将nii转为图片后用传统cnn方法做的，而baseline为直接读取nii做的，另外可以尝试三维卷积，据说三维卷积不用trick也可以跑到0.71多
