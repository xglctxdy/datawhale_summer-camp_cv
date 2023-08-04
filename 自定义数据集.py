class myDataset(Dataset):  # 自定义数据集类，里面包含了图像剪裁操作
    def __init__(self, img_path, transform=None):
        self.img_path = img_path
        if transform is not None:
            self.transform = transform
        else:
            self.transform = None

    @staticmethod
    def fillcrop(img):  # 把第一次裁剪的图片填充为正方形，避免resize时会改变比例
        img = np.array(img)

        # 以最长的一边为边长，把短的边补为一样长，做成正方形，避免resize时会改变大脑的比例
        down = img.shape[0]  # 行数
        up = img.shape[1]  # 列数
        max1 = max(down, up)  # 比较行数和列数哪个大，来决定以谁作为正方形的边长
        down = (max1 - down) // 2  # 整除，大的值必然为0，而小的需要补上相应的值。比方说down更大，则赋值以后down为0，表示行不需要填充像素点
        up = (max1 - up) // 2
        down_zuo, down_you = down, down
        up_zuo, up_you = up, up

        if (max1 - img.shape[0]) % 2 != 0:  # 考虑奇数的情况，若为奇数则需要在行的左边多加一列
            down_zuo = down_zuo + 1
        if (max1 - img.shape[1]) % 2 != 0:
            up_zuo = up_zuo + 1

        matrix_pad = np.pad(img, pad_width=((down_zuo, down_you),  # 向上填充1个维度，向下填充两个维度
                                            (up_zuo, up_you),  # 向左填充2个维度，向右填充一个维度
                                            (0, 0)),  # 通道维度不填充
                            mode="constant",  # 填充模式
                            constant_values=(0, 0))  # 第一个维度（就是向上和向左）填充6，第二个维度（向下和向右）填充5
        return matrix_pad

    def precrop(self, image_data):  # 初次裁剪大脑
        scaled_data = (image_data - np.min(image_data)) / (np.max(image_data) - np.min(image_data)) * 255
        img = np.uint8(scaled_data)

        index = np.where(img > 50)  # 找出像素值大于50的所以像素值的坐标，也就是获取表示大脑部分的像素的索引
        x = index[0]  # 获取所有表示大脑部分的像素的横坐标
        y = index[1]  # 获取所有表示大脑部分的像素的纵坐标
        max_x = max(x)  # 横坐标的最大值表示有边界，其余同理
        min_x = min(x)
        max_y = max(y)
        min_y = min(y)
        max_x = max_x + 10  # +10是为了多裁剪一部分黑色的区域，防止可以有部分边缘的大脑未被裁剪
        min_x = min_x - 10
        max_y = max_y + 10
        min_y = min_y - 10
        if max_x > img.shape[0]:
            max_x = img.shape[0]
        if min_x < 0:
            min_x = 0
        if max_y > img.shape[1]:
            max_y = img.shape[1]
        if min_y < 0:
            min_y = 0
        img = img[min_x:max_x, min_y:max_y, :]  # 保留全部的通道维度
        return self.fillcrop(img)

    def __getitem__(self, index):
        if self.img_path[index] in DATA_CACHE:
            img = DATA_CACHE[self.img_path[index]]
        else:  # 从磁盘中读取图像，并缓存到DATA_CACHE中
            img = nib.load(self.img_path[index])
            img = img.dataobj[:, :, :, 0]
            DATA_CACHE[self.img_path[index]] = img

        # img_fdata = img.get_fdata()
        (_, _, z) = img.shape
        # idx = np.random.choice(range(img.shape[-1]), 60)
        img = img[:, :, int(z / 2 - 20):int(z / 2 + 20)]
        img = img.astype(np.float32)
        img = self.precrop(img)

        if self.transform is not None:
            img = self.transform(image=img)['image']

        img = img.transpose([2, 0, 1])  # 将原始的3维numpy数组的维度从(height, width, depth)变成了(depth, height, width)
        label = int('NC' in self.img_path[index])
        return img.astype(np.float32), torch.tensor(label, dtype=torch.long)

    def __len__(self):
        return len(self.img_path)