## 基于深度学习框架darknet的yolov4

### [参考资料](https://github.com/AlexeyAB/darknet)


### 使用步骤
#### 1.下载darknet包

> `>> git clone https://github.com/AlexeyAB/darknet.git`


#### 2.下载pre-trained weights

> `>> wget https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.conv.137`


#### 3.在/darknet/data文件夹下创建pic和xml这两个文件夹


#### 4.将训练图像存放到/darknet/data/pic文件夹下,将对应图像名的对应xml文件存放到/darknet/data/xml文件夹下

> xml文件获得方法使用 **[labelImg](https://github.com/tzutalin/labelImg)** 工具


#### 5.下载两个文件yolov4run.py和configer.py到/darknet目录下

> [下载地址](https://github.com/GitZzw/Study_notes/tree/master/YOLOv4)


#### 6.运行yolov4run.py,然后根据提示操作即可

> `>> python yolov4run.py`



### 说明

#### 总共分为四种模式，训练，测试，继续训练和验证
   
>   测试：测试某张图像的识别结果并输出图形
   
>   继续训练：使用之前的权重文件继续训练
   
>   验证：用训练好的权重文件计算验证集的识别准确率

>  训练权重结果文件每迭代100次会自动保存到/darknet/backup/文件夹下

>  停止迭代次数的选择可以[参考内容](https://github.com/AlexeyAB/darknet#when-should-i-stop-training)

![图片](https://camo.githubusercontent.com/51af5be5cfa94b6d741c90d10a163b168bf9170e/68747470733a2f2f6873746f2e6f72672f66696c65732f3564632f3761652f3766612f35646337616537666164396434653365623361343834633538626663316666352e706e67)
