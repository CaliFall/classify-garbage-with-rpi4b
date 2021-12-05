# 基于树莓派的智能垃圾分拣系统
# Intelligent garbage sorting system based on raspberry pi 4b

#### 介绍/Intro
>基于树莓派4b的智能垃圾分拣系统，涉及pytorch，pyqt5。
>The intelligent garbage sorting system based on raspberry pie 4B involves pytorch and pyqt5.
>为第七届工训赛设计，获江苏省特等奖，第五名无缘国赛。
>It was designed for the seventh engineering training competition, won the special prize of Jiangsu Province and missed the national competition.
>也可以当作毕设。
>It can also be used as a graduation project.
>Bilibili演示视频:https://www.bilibili.com/video/BV1kN411o73u
>BiliBili demo video:https://www.bilibili.com/video/BV1kN411o73u


#### 联系方式/How to get in touch with me?
>QQ交流群群号:789169244
>QQ communication group number:789169244
>我(何天骅)QQ号:3177556879
>My (CaliFall) QQ number:3177556879
>Or just email me:califall@qq.com


#### 开发人员/Developer
>软件部分:南京工程学院 自动化学院 机器人工程 20级大一 何天骅（主要开发人员） || 南京工程学院 20级研一 岳骏 (副开发人员)
>
>机械结构:南京工程学院 工业中心 20级大一 张毅（主要开发人员） || 南京工程学院 机械工程学院 20级大一 徐鸣远（副开发人员）


#### 文件组成/File Usage
>##### data_set文件夹(Folder)
>>###### train文件夹(Folder)
>>>模型训练集(Training set)
>>###### val文件夹(Folder)
>>>模型验证集(Validation set)

>##### logs文件夹(Folder)
>>用于存放程序运行时产生的日志，如果被删除了需要手动创建一个
>>The validation set is used to store the logs generated when the program is running. If they are deleted, you need to create one manually

>##### photo文件夹(Folder)
>>用于存放程序运行时拍摄的照片，如果被删除了需要手动创建一个
>>It is used to store the photos taken when the program is running. If they are deleted, you need to manually create one

>##### tools文件夹(Folder)
>>###### model.py
>>>本质上是mobilenet_v2，用于训练时调用
>>>It's essentially mobilenet_ V2, used to call during training
>>###### predict.py
>>>测试模型用的程序
>>>Program for testing model
>>###### train.py
>>>训练模型用的程序
>>>Program for training model

>##### weight文件夹(Folder)
>>###### class_indices.json
>>>训练模型时自动生成的词典
>>>Dictionary automatically generated when training model

>>###### xxx.pth
>>>训练好的模型
>>>Trained model

>##### alarm.py
>>控制蜂鸣器的函数
>>Function controlling buzzer

>##### getpic.py
>>拍摄训练图片用的程序
>>Procedures for taking training pictures

>##### main.py
>>垃圾分拣流程，注意:单独运行这个文件没有UI
>>Garbage sorting process, note: there is no UI for running this file alone

>##### motocontrol.py
>>控制舵机的函数
>>Function for controlling steering motor

>##### sensor.py
>>读取超声波传感器的函数
>>Function for read ultrasonic sensor

>##### ui_g.py
>>UI文件的图形部分，是ui文件用PYUIC转换然后修改过的
>>The graphic part of the UI file ,which is converted and modified by pyuic
>
>>如果你重新生成了此文件，把PYQT5自动生成的代码即最后一行删掉，然后自己加上必要的库
>>If you regenerate this file, delete the last line of the code automatically generated by pyqt5, and then add the necessary libraries yourself

>##### ui_l.py
>>UI文件的逻辑部分
>>Logical part of UI

>##### ui.ui
>>UI原文件
>>UI original file(Of course it's Chinese beacause I speak Chinese)

#### 电气原理图/Schematic diagram
![电气原理图](https://images.gitee.com/uploads/images/2021/0501/131846_afd4c5d9_8347966.png "yuanlitu.png")

#### UI预览图/UI Preview
![输入图片说明](https://images.gitee.com/uploads/images/2021/1015/193533_634f978e_8347966.png "ui界面预览图.png")

#### 零件表/Parts list
>1. 树莓派4b (用的4g运存版,2g应该也行)
>1. Raspberry pie 4B (2G should be OK,while I ran it on a 4G storage version)
>2. 扩展板 (YwRobot家的,35rmb,不是广告)
>2. Expansion board (ywrobot's, 35rmb, not advertising)
>   (btw,i buy it on taobao.com,I think you can get a alternative wherever you are)
>3. 舵机 x 2 (我们用的机构里要两个舵机,由于一个是360度一个是180度,所以代码写的有些奇怪)
>3. Steering gear x 2 (we need two steering gears in the mechanism. Because one is 360 degrees and the other is 180 degrees, the code is strange)
>4. 超声波传感器 x 4 (HC-SRO4,四针的) (建议不要用超声波,稳定性太差)
>4. Ultrasonic sensor x 4 (hc-sro4, four pin) (it is recommended not to use ultrasonic, the stability is too poor)
>5. 显示屏 (至少要1024x600的分辨率,否则程序界面要大改)
>5. Display screen (at least 1024x600 resolution, otherwise the program interface will be greatly changed)
>6. 开关 x 6 (三个电路三个控制) (放这么多开关是老师的主意,后续来看确实有这个必要) (如果没有开关,则需要把空闲模式关掉)
>6. Switch x 6 (three circuits and three controls) (it is the teacher's idea to put so many switches, which is really necessary later) (if there is no switch, it is necessary to turn off the idle mode)

#### 相关图片/Related pictures
>##### 大致走线图/Approximate routing diagram
>>![大致走线图](https://images.gitee.com/uploads/images/2021/0503/192743_9ada2d26_8347966.png "3.png")
>##### 工训赛社区赛现场/Site of work training competition and Community Competition
>>![工训赛社区赛现场](https://images.gitee.com/uploads/images/2021/0503/192651_54492313_8347966.png "2.png")
>##### 垃圾桶外观图/Appearance drawing of trash can
>>![垃圾桶外观图](https://images.gitee.com/uploads/images/2021/0503/192506_8f03823e_8347966.png "1.png")
>##### 校内媒体图片/School media pictures
>>![校内媒体图片](https://images.gitee.com/uploads/images/2021/0503/192556_7e0475dc_8347966.jpeg "mmexport1620040617070.jpg")
>##### 电视台采访图片/TV interview pictures
>>![电视台采访图片](https://images.gitee.com/uploads/images/2021/0503/192417_5917d487_8347966.png "9ROI9K2PJHH_04}%U$THDLW(1).png")
