# -*- coding: utf-8 -*-
## 配置文件

import os
class configer(object):
    def __init__(self,mode=0,classlist=['']):
        self.modelist = ['train', 'test', 'retrain','valid']

        self.mode = self.modelist[mode]  

        self.classes = classlist


        # 这只是部分常用参数设置，具体设置在.cfg文件中
        # 参数含义参考 https://www.cnblogs.com/shierlou-123/p/11152623.html
        self.filters = (len(self.classes) + 5) * 3
        self.batch = 64
        self.subdivisions = 16
        self.max_batches = max(6000,len(self.classes)*2000)
        self.width = 416
        self.height=416
        self.classnum = len(self.classes)
    


        ##########################请勿随意更改下列设置##############################


        self.darknet_path = os.getcwd()
        self.cfg_path = os.path.join(self.darknet_path, 'yolov4_own.cfg')
        # 训练支撑文件地址
        self.voc_data_path = os.path.join(self.darknet_path, 'data/own.data')  
        self.voc_names_path = os.path.join(self.darknet_path, 'data/own.names')



        # pic和xml地址 需要绝对路径
        self.pic_path = os.path.join(self.darknet_path, 'data/pic')
        self.xml_path = os.path.join(self.darknet_path, 'data/xml')
        self.txt_path = os.path.join(self.darknet_path, 'data/txt')
        self.labels_path = os.path.join(self.darknet_path, 'pic')
        pic_type = os.listdir(self.pic_path)[0].split('.')[-1]
        self.pic_type = pic_type  # 图片格式 jpg,png,其他图片格式均可
