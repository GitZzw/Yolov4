# -*- coding: utf-8 -*-
#　main函数
import os
import sys
import shutil
from configer import configer
import time
import xml.etree.ElementTree as ET
import pickle


#训练
class train_yolo():
    def __init__(self, config):
        self.config = config
    
    #检查图片对应的xml文件是否存在
    def file_check(self):
        img_total = os.listdir(self.config.pic_path)
        for i in img_total:
            img_name = i.split('.')[0]
            xml = os.path.join(self.config.xml_path, str(img_name) + '.xml')
            if not os.path.exists(xml):
                pic_path = os.path.join(self.config.pic_path, i)
                os.remove(pic_path)
                print(img_name, '.xml文件不存在:', ' 我已经把这张图删了')

    #修改训练所需的voc_data.txt文件
    #其中包含训练分类的数量,训练集路径，验证集路径,训练名字分类文件voc_name.txt路径，生成权重存放路径
    def change_voc_data(self):
        text = []
        text.append('classes = {}'.format(len(self.config.classes)))
        text.append('train = {}'.format(os.path.join(self.config.darknet_path, 'data/train.txt')))
        text.append('valid = {}'.format(os.path.join(self.config.darknet_path, 'data/valid.txt')))
        text.append('names = {}'.format(self.config.voc_names_path))
        text.append('backup = {}'.format(os.path.join(self.config.darknet_path,'backup')))
        with open(self.config.voc_data_path, 'w') as f:
            for i in range(len(text)):
                f.write(text[i] + '\n')


    #修改训练所需的voc_names.txt文件
    #包含类别名称，e.g. target pickup
    def change_voc_names(self):
        with open(self.config.voc_names_path, 'w') as f:
            for i in range(len(self.config.classes)):
                f.write(self.config.classes[i] + '\n')
    
    #修改cfg文件
    #主要对神经网络等参数进行调整
    def change_cfg(self):
        with open(self.config.cfg_path, 'r+') as f:
            cfg = f.readlines()
            cfg[5] = 'batch={}\n'.format(self.config.batch)
            cfg[6] = 'subdivisions = {}\n'.format(self.config.subdivisions)
            cfg[19] = 'max_batches={}\n'.format(self.config.max_batches)

            cfg[7] = 'width={}\n'.format(self.config.width)
            cfg[8] = 'height={}\n'.format(self.config.height)

            cfg[962] = 'filters={}\n'.format(self.config.filters)
            cfg[969] = 'classes={}\n'.format(self.config.classnum)

            cfg[1050] = 'filters={}\n'.format(self.config.filters)
            cfg[1057] = 'classes={}\n'.format(self.config.classnum)

            cfg[1138] = 'filters={}\n'.format(self.config.filters)
            cfg[1145] = 'classes={}\n'.format(self.config.classnum)
        with open(self.config.cfg_path, 'w') as f:
            for lines in cfg:
                f.write(lines)


    #xml文件转换为包含有label坐标的txt文件
    def covert_to_txt(self):

        total_xml = os.listdir(self.config.xml_path)
        ftrain = open('%s/train.txt' % (self.config.txt_path), 'w')
        for i in range(len(total_xml)):
            name = total_xml[i][:-4] + '\n'
            ftrain.write(name)
        ftrain.close()

        

        image_ids = open('{}/train.txt'.format(self.config.txt_path)).read().strip().split()
        list_file = open('%s/data/train.txt'%(self.config.darknet_path),'w')
        for image_id in image_ids:
            current_pic_path = os.path.join(self.config.pic_path,str(image_id)+'.'+self.config.pic_type+'\n')
            list_file.write(current_pic_path)
            self.convert_annotation(image_id)
        list_file.close()

        shutil.copy('%s/data/train.txt' % (self.config.darknet_path), '%s/data/valid.txt' % (self.config.darknet_path))

    #xml文件转换为包含有label坐标的txt文件    
    def convert(self,size, box):
        dw = 1./size[0]
        dh = 1./size[1]
        x = (box[0] + box[1])/2.0
        y = (box[2] + box[3])/2.0
        w = box[1] - box[0]
        h = box[3] - box[2]
        x = x*dw
        w = w*dw
        y = y*dh
        h = h*dh
        return (x,y,w,h)
    
    #xml文件转换为包含有label坐标的txt文件
    def convert_annotation(self,image_id):
        curent_xml_path = os.path.join(self.config.xml_path,str(image_id)+'.xml')
        in_file = open(curent_xml_path,'r')
        out_file = open('%s/%s.txt'%(self.config.pic_path,image_id), 'w')
        tree=ET.parse(in_file)
        root = tree.getroot()
        size = root.find('size')
        w = int(size.find('width').text)
        h = int(size.find('height').text)

        for obj in root.iter('object'):
            difficult = obj.find('difficult').text
            cls = obj.find('name').text
            if cls not in self.config.classes or int(difficult) == 1:
                continue
            cls_id = self.config.classes.index(cls)
            xmlbox = obj.find('bndbox')
            b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
            bb = self.convert((w,h), b)
            out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')


#验证
class valid_yolo():
    def __init__(self,config,validpicpath,validxmlpath):
        self.config = config
        self.validpicpath = validpicpath
        self.validxmlpath = validxmlpath


    def file_check(self):
        command = "rm -rf {}/*.txt".format(self.validpicpath)
        os.system(command)
        img_total = os.listdir(self.validpicpath)
        for i in img_total:
            img_name = i.split('.')[0]
            xml = os.path.join(self.validxmlpath, str(img_name) + '.xml')
            if not os.path.exists(xml):
                pic_path = os.path.join(self.validpicpath, i)
                os.remove(pic_path)
                print(img_name, '.xml文件不存在:', ' 我已经把这张图删了')

    #自定义验证图像及xml路径，生成包含对应label坐标的txt文件
    def custom(self):

        piclist = os.listdir(self.validpicpath)
        fvalid = open('%s/data/valid.txt' % (self.config.darknet_path), 'w')
        for i in range(len(piclist)):
            name = self.validpicpath + '/' + piclist[i] + '\n'
            fvalid.write(name)
            print(piclist[i][:-4])
            self.convert_annotation(piclist[i][:-4])
        fvalid.close()

    def convert(self,size, box):
        dw = 1./size[0]
        dh = 1./size[1]
        x = (box[0] + box[1])/2.0
        y = (box[2] + box[3])/2.0
        w = box[1] - box[0]
        h = box[3] - box[2]
        x = x*dw
        w = w*dw
        y = y*dh
        h = h*dh
        return (x,y,w,h)

    def convert_annotation(self,index):
        in_file = open(self.validxmlpath + '/'+ index +'.xml','r')
        print(self.validxmlpath + '/'+ index +'xml')
        out_file = open('%s/%s.txt'%(self.validpicpath,index), 'w')
        tree=ET.parse(in_file)
        root = tree.getroot()
        size = root.find('size')
        w = int(size.find('width').text)
        h = int(size.find('height').text)

        for obj in root.iter('object'):
            difficult = obj.find('difficult').text
            cls = obj.find('name').text
            if cls not in self.config.classes or int(difficult) == 1:
                continue
            cls_id = self.config.classes.index(cls)
            xmlbox = obj.find('bndbox')
            b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
            bb = self.convert((w,h), b)
            out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')












#训练
def yolo_train(config):

    # 文件检查 保证xml文件以及对应的pic文件存在
    train = train_yolo(config)

    train.file_check()

    # 生成train.txt,valid.txt,test.txt文件，txt中每一行是对应pic数据集的路径
    # 由xml文件生成对应pic文件的label位置，并放在同名txt文件中(e.g. 1.png -> 1.txt    2.png -> 2.txt )
    train.covert_to_txt()

    


    #生成训练需要的voc.data和voc.names文件
    train.change_voc_data()
    train.change_voc_names()

    current_path = os.getcwd()
    # 确认是否有yolo4.conv文件
    if(not os.path.exists('yolov4_own.cfg')):
        #从costom_cfg文件拷贝为yolo_obj.cfg文件
        shutil.copy(current_path+str('/cfg/yolov4-custom.cfg'), current_path+str('/yolov4_own.cfg'))


    else:
        pass


    #修改cfg文件配置
    train.change_cfg()  


    #创建backup文件夹
    if(not os.path.exists('backup')):
        #从costom_cfg文件拷贝为yolo_obj.cfg文件
        os.system('mkdir ./backup/')

    else:
        pass


    command = "./darknet detector train {} {} yolov4.conv.137".format(config.voc_data_path,config.cfg_path)
    os.system(command)


#测试一张图像
def yolo_test(config):
    if(not os.path.exists('backup')):
        print('没有权重文件，请重新训练')
        exit(0)
    else:
    
        lists = os.listdir('./backup')
        lists.sort(key=lambda x:os.path.getmtime('./backup' +'//'+x))
        weights_new = os.path.join('./backup',lists[-1])
        print('use weights file named ' + weights_new +' to test')
    
    testpath = input('<<<<<<<请输入要测试的图片存放文件夹路径<<<<<<<\n')
    while(not os.path.exists(testpath)):
        testpath = input('<<<<<该图片文件不存在，请检查路径并重新输入<<<<<<<\n')

    command = "./darknet detector test {} {} {} -ext_output {}".format(config.voc_data_path,config.cfg_path,weights_new,testpath)

    os.system(command)




#继续之前的权重训练
def yolo_retrain(config):
    weights_retrain = input('<<<<<<<<<<<<<<<<<请输入上次未训练完的weight路径,一般保存在darknet/backup文件夹下<<<<<<<<<<<<\n')
    while(not os.path.exists(weights_retrain)):
    	weights_retrain = input('文件不存在，请检查重新输入\n')
    command = "./darknet detector train {} {} -ext_output {}".format(config.voc_data_path,config.cfg_path,weights_retrain)

    os.system(command)



#验证
def yolo_valid(config):
    if(not os.path.exists('backup')):
        print('没有权重文件，请训练')
        exit(0)
    else:
        ans = input('<<<<<<将训练集作为验证集输入 yes ,自定义验证集输入 no <<<<<<<<<<<<<<\n')
        while(ans != 'yes' and ans != 'no'):
            ans = input('<<<<<<错误输入，请重新输入<<<<<<<<<<<<<<\n')
        if(ans == 'yes'):
            shutil.copy('%s/data/train.txt' % (config.darknet_path), '%s/data/valid.txt' % (config.darknet_path))
            lists = os.listdir('./backup')
            lists.sort(key=lambda x:os.path.getmtime('./backup' +'/'+x))
            weights_valid = os.path.join('./backup',lists[-1])
            command = "./darknet detector map {} {} {}".format(config.voc_data_path,config.cfg_path,weights_valid)

            os.system(command)
        else:
            validpicpath = input('<<<<<<<<<<请输入验证图像路径<<<<<<<<<<<<<<<<<<\n')
            while(not os.path.exists(validpicpath)):
                validpath = input('<<<<<<<<<<图像路径有误，重新输入<<<<<<<<<<<<<<<<<<\n')
            validxmlpath = input('<<<<<<<<<<继续输入对应xml文件路径，将自动生成验证集<<<<<<<<<<<<<<<<<<\n')
            while(not os.path.exists(validxmlpath)):
                validpath = input('<<<<<<<<<<xml路径有误，重新输入<<<<<<<<<<<<<<<<<<\n')
            valid=valid_yolo(config,validpicpath,validxmlpath)
            valid.file_check()
            valid.custom()

            print('验证集已生成，开始验证')
            time.sleep(1)
            lists = os.listdir('./backup')
            lists.sort(key=lambda x:os.path.getmtime('./backup' +'/'+x))
            weights_valid = os.path.join('./backup',lists[-1])
            command = "./darknet detector map {} {} {}".format(config.voc_data_path,config.cfg_path,weights_valid)

            os.system(command)






#输入模式
def getinput():
    mode = int(input('<<<<<<请输入模式  (train输入0;  test输入1;  retrain输入2;  valid输入3)<<<<<<<<\n'))
    while(mode != 0 and mode!=1 and mode!=2 and mode!=3):
        print('input wrong,again')
        mode = int(input('<<<<<<请输入模式  (train输入0;  test输入1;  valid输入2;  valid输入3)<<<<<<<<\n'))
    return mode
#输入分类名称
def getclasslist():
    classstr = input("<<<<请输入分类的目标集,以空格分隔,回车结束<<<<<<<<<\n")
    classlist=classstr.split(' ')
    message = 'train mode and classlist is [' + ','.join(classlist) + '],press c to continue or press other key to re-input\n'
    exam = input(message)
    return [exam,classlist]





#环境配置
def circumstance():
    if(not os.path.exists('yolov4.conv.137')):
        print('缺少文件，准备下载pre-trained weights')
        command = "wget https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.conv.137"
        os.system(command)
    else:
        pass
    if(not os.path.exists('data/pic/')):
        print('需要将训练集照片按要求放置到data/pic/文件夹')
    else:
        pass
    if(not os.path.exists('data/txt/')):
        com = 'mkdir ./data/txt/'
        os.system(com)
    else:
        pass
    if(not os.path.exists('darknet')):
        print('<<<<<<<<<<<<<<<<<<<正在编译darknet包<<<<<<<<<<<<<<<<<<<<<<')
        print('<<<<<<<<<<<<<<<<<<<正在编译darknet包<<<<<<<<<<<<<<<<<<<<<<')
        print('<<<<<<<<<<<<<<<<<<<正在编译darknet包<<<<<<<<<<<<<<<<<<<<<<')
        time.sleep(2)
        command1 = './build.sh'
        os.system(command1)
    else:
    	pass
    #  开启GPU加速
    with open('Makefile') as f:
            mk = f.readlines()
            mk[0] = 'GPU=1\n'
            mk[1] = 'CUDNN=1\n'
            mk[3] = 'OPENCV=1\n'
    with open('Makefile', 'w') as f:
            for lines in mk:
                f.write(lines)






if __name__ == '__main__':

    
    #环境配置
    circumstance()
    # 获取输入 判断四种模式 
    mode = getinput()

    if (mode == 0):
        command1 = "rm -rf ./data/pic/*.txt"
        os.system(command1)
        exam,classlist = getclasslist()
        while(exam != 'c'):
            exam,classlist = getclasslist()
        config = configer(0,classlist)
        yolo_train(config)
    

    elif(mode == 1):
        config = configer(1)

        yolo_test(config)
    


    elif(mode == 2):
        config = configer(2)
        yolo_retrain(config)
        
    elif(mode == 3):
    	config = configer(3)
    	yolo_valid(config)
