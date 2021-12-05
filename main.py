# -*- coding: utf-8 -*-

import time  # 时间库
import cv2  # OpenCV库
import re  # 正则库
import numpy as np  # 哈希算法涉及

import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import json
from torchvision.models import MobileNetV2
import motocontrol as mc
import alarm


# 静态变量类
class Static:
    COUNT = 0

    def __init__(self, arg):
        self.arg = arg
        Static.COUNT += 1


# 整合之后的分拣代码
class DetectMain(Static):
    # 初始化函数(类在启动时会自动运行一遍)
    def __init__(self):
        self.model_version = 2.6    # 当前模型版本
        Static(1)  # 设置静态变量
        # 垃圾分拣相关函数初始化(程序启动时会运行一次)
        if Static.COUNT == 1:
            self.notloop()
            Static(0)
        # 循环部分
        self.isloop("fir")

    # 不参与循环部分:
    def notloop(self):
        # 全局声明
        global detect_time
        global hash_limit
        global CAM_NUM

        global full_flag_1
        global full_flag_2
        global full_flag_3
        global full_flag_4

        #摄像头参数
        CAM_NUM = 0

        # 初始化统计日志
        self.makecount(0)

        # 初始化检测次数变量
        # 初始为0,勿改
        detect_time = 0

        # 自动垃圾检测灵敏度(理论上数值越高越灵敏)
        # 理论取值区间为 0~1.0
        hash_limit = 0.6

        # 拍摄函数注释:
        # capture(1):手动拍摄垃圾
        # capture(2):自动拍摄垃圾
        # capture_hash(1):手动拍摄垃圾托盘(空)
        # capture_hash(2):自动拍摄垃圾托盘(空)
        # capture_hash(3):手动拍摄垃圾托盘(未知)
        # capture_hash(4):自动拍摄垃圾托盘(未知)
        # 一般来说手动模式在实际运行时并不会用到

        print(torch.__version__)
        print("初始化完毕")

    # 参与循环部分:
    def isloop(self, hashflag):
        print(hashflag)
        global HASH1
        global detect_time
        global name_max_final
        global cate_max_final

        if hashflag == "fir":
            # 初始化垃圾托盘图片
            time.sleep(0.5)                                 # 延时拍摄防止托盘抖动
            self.capture_hash(2)                            # 拍摄托盘图片
            HASH1 = self.pHash("./photo/platform.jpg")      # 生成托盘哈希值
            print("托盘照片更新")

            self.readcond()                                 # 读取满载日志

        # 初始化最终结果
        name_max_final = 'null'
        cate_max_final = ''

        detect_time += 1
        
        alarm.warningNope()                         # 蜂鸣器提示检测托盘
        out_score = self.hash_score(detect_time)    # 返回相似度
        
        print("ddmode=", self.ddmode)          # 打印双重投放开关
        print("ddflag=", self.ddflag)          # 打印双重投放标识

        if (not self.ddmode and out_score >= hash_limit) or (self.ddmode and self.ddflag == "A" and out_score >= hash_limit):
            print("等待垃圾中...")
            self.isloop("mul")  # 嵌套

        else:
            self.time_total(1)  # 开始计时

            print("检测到垃圾!")

            # 双重投放额外延时1秒
            if self.ddmode and self.ddflag == "A":
                time.sleep(1)
                
            alarm.warningCam()
            
            self.capture(2)     # 初次捕获照片
            self.check()        # 检测垃圾

            # 双重投放第二次时先复位再等待一秒
            if self.ddmode and self.ddflag == "B":
                mc.motoact(0)
                time.sleep(1)

            # 打印最可能的可信度、垃圾名、垃圾类别
            print("cate_max_final=", cate_max_final)

            # 测试时暂时关闭分拣功能
            self.connection(self.predict_cla)
            self.time_total(0)  # 打印完整检测一个垃圾的时间

            # 生成本次检测的日志文件
            self.makelog()

    # 拍摄垃圾
    def capture(self, ind):

        # 手动拍摄垃圾
        if ind == 1:
            cap = cv2.VideoCapture(CAM_NUM)
            while (1):
                ret, frame = cap.read()
                cv2.imshow("capture", frame)

                if cv2.waitKey(1) & 0xFF == ord('p'):
                    cv2.imwrite("./photo/garbage.jpg", frame)
                    break
            cap.release()
            cv2.destroyAllWindows()

        # 自动拍摄垃圾
        elif ind == 2:
            cap = cv2.VideoCapture(CAM_NUM)
            ret, frame = cap.read()
            cv2.imwrite("./photo/garbage.jpg", frame)
            cap.release()

        else:
            print("拍照Ⅰ参数错误!")

    # 拍照检测垃圾是否投入
    def capture_hash(self, ind):
        # 手动拍摄垃圾托盘(空)
        if ind == 1:
            cap = cv2.VideoCapture(CAM_NUM)
            while True:
                ret, frame = cap.read()
                cv2.imshow("capture", frame)
                if cv2.waitKey(1) & 0xFF == ord('p'):
                    cv2.imwrite("./photo/platform.jpg", frame)
                    break
            cap.release()
            cv2.destroyAllWindows()

        # 自动拍摄垃圾托盘(空)
        elif ind == 2:
            cap = cv2.VideoCapture(CAM_NUM)
            ret, frame = cap.read()
            cv2.imwrite("./photo/platform.jpg", frame)
            # cap.release()

        # 手动拍摄垃圾托盘(未知)
        elif ind == 3:
            cap = cv2.VideoCapture(CAM_NUM)
            while True:
                ret, frame = cap.read()
                cv2.imshow("capture", frame)

                if cv2.waitKey(1) & 0xFF == ord('p'):
                    cv2.imwrite("./photo/garbage_hash.jpg", frame)
                    break
            cap.release()
            cv2.destroyAllWindows()

        # 自动拍摄垃圾托盘(未知)
        elif ind == 4:
            cap = cv2.VideoCapture(CAM_NUM)
            ret, frame = cap.read()
            cv2.imwrite("./photo/garbage_hash.jpg", frame)
            cap.release()

        else:
            print("拍照Ⅱ参数错误!")

    # 推断垃圾
    def check(self):
        # 模型位置
        model_weight_path = "./weight/mobilenet_v2_Ori_" + str(self.model_version) + ".pth"

        global cate_max_final
        data_transform = transforms.Compose(
            [transforms.Resize(256),
             transforms.CenterCrop(224),
             transforms.ToTensor(),
             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

        # load image
        img = Image.open("./photo/garbage.jpg")
        plt.imshow(img)
        # [N, C, H, W]
        img = data_transform(img)
        # expand batch dimension
        img = torch.unsqueeze(img, dim=0)
        # print("img:", img)
        # read class_indict
        try:
            json_file = open('./weight/class_indices.json', 'r')
            class_indict = json.load(json_file)
        except Exception as e:
            print(e)
            exit(-1)
        # create model
        model = MobileNetV2(num_classes=26)
        # load model weights
        model.load_state_dict(torch.load(model_weight_path, map_location=torch.device('cpu')))

        model.eval()
        with torch.no_grad():
            # predict class
            output = torch.squeeze(model(img))
            predict = torch.softmax(output, dim=0)
            predict_cla = torch.argmax(predict).numpy()
        print("predict_class="+str(predict_cla))
        self.predict_cla = int(predict_cla)

        if self.predict_cla >= 0 and self.predict_cla <= 5:
            cate_max_final = "可回收垃圾"
            self.detail = self.predict_cla
            self.predict_cla = 1
        elif self.predict_cla >= 6 and self.predict_cla <= 11:
            cate_max_final = "有害垃圾"
            self.detail = self.predict_cla
            self.predict_cla = 2
        elif self.predict_cla >= 12 and self.predict_cla <= 19:
            cate_max_final = "厨余垃圾"
            self.detail = self.predict_cla
            self.predict_cla = 3
        elif self.predict_cla >= 20 and self.predict_cla <= 25:
            cate_max_final = "其他垃圾"
            self.detail = self.predict_cla
            self.predict_cla = 4

    # 读取满载情况以及双重投放状态
    def readcond(self):
        global full_flag_1
        global full_flag_2
        global full_flag_3
        global full_flag_4

        with open('./logs/cond.txt', 'r', encoding='utf-8') as f:
            cond = f.read()
            pattern_cond = re.compile('full_flag=(.*?)\n')
            pattern_ddmode = re.compile('ddmode=(.*?)\n')
            pattern_ddflag = re.compile('ddflag=(.*?)\n')

            list_cond = pattern_cond.findall(cond)
            ddmode = pattern_ddmode.findall(cond)
            ddflag = pattern_ddflag.findall(cond)

        full_flag_1 = list_cond[0]
        full_flag_2 = list_cond[1]
        full_flag_3 = list_cond[2]
        full_flag_4 = list_cond[3]

        self.ddmode = ddmode[0]
        self.ddflag = ddflag[0]
        
        if self.ddmode == "True":
            self.ddmode = True
        else:
            self.ddmode = False

    # 生成垃圾信息日志文件
    def makelog(self):
        with open('./logs/log.txt', 'w+', encoding='utf-8') as f:
            f.write('cate_max_final=' + cate_max_final + '\n')
            f.write('cate_detail=' + str(self.detail) + '\n')
            f.write('time_total=' + format(time_end - time_start, '.3f') + 's' + '\n')
        print("信息日志已生成")

    # 生成垃圾计数日志文件
    def makecount(self, ind):
        # 参数为0时是初始化
        if ind == 0:
            global count_list
            count_list = [0, 0, 0, 0, 0]
        # 参数不为0时对应元素加一
        else:
            count_list[0] += 1
            count_list[ind] += 1

        # 把列表存到文件中
        with open('./logs/list.txt', 'w+', encoding='utf-8') as f:
            f.write('totalnum=' + str(count_list[0]) + '\n')
            f.write('cate1num=' + str(count_list[1]) + '\n')
            f.write('cate2num=' + str(count_list[2]) + '\n')
            f.write('cate3num=' + str(count_list[3]) + '\n')
            f.write('cate4num=' + str(count_list[4]) + '\n')
        print("计数日志已生成")

    # 衔接函数
    def connection(self, mode):

        print("connection=", mode)
        if mode == 1:                           # 模式代码
            if full_flag_1 == "True":           # 如果已满
                mc.motoact(7)                   # 舵机警告
                alarm.warning()                 # 蜂鸣器警报
                time.sleep(3)                   # 延时三秒,取出垃圾
            elif self.ddmode:                   # 如果双重投放模式打开
                mc.motoact_dd(1, self.ddflag)   # 舵机双重投放动作
            else:
                mc.motoact(1)                   # 舵机普通投放运动
                self.makecount(1)               # 计数累加

        if mode == 2:
            if full_flag_2 == "True":
                mc.motoact(7)
                alarm.warning()
                time.sleep(3)
            elif self.ddmode:
                mc.motoact_dd(2, self.ddflag)
            else:
                mc.motoact(2)  # 舵机运动
                self.makecount(2)  # 计数累加

        if mode == 3:
            if full_flag_3 == "True":
                mc.motoact(7)
                alarm.warning()
                time.sleep(3)
            elif self.ddmode:
                mc.motoact_dd(3, self.ddflag)
            else:
                mc.motoact(3)  # 舵机运动
                self.makecount(3)  # 计数累加

        if mode == 4:
            if full_flag_4 == "True":
                mc.motoact(7)
                alarm.warning()
                time.sleep(3)
            elif self.ddmode:
                mc.motoact_dd(4, self.ddflag)
            else:
                mc.motoact(4)  # 舵机运动
                self.makecount(4)  # 计数累加

    # 感知哈希检测部分
    def pHash(self, imgfile):
        # 求哈希值的一个东西(看不大懂)

        img_list = []
        # 加载并调整图片为32x32灰度图片
        img = cv2.imread(imgfile, 0)
        img = cv2.resize(img, (64, 64), interpolation=cv2.INTER_CUBIC)
        # 创建二维列表
        h, w = img.shape[:2]
        vis0 = np.zeros((h, w), np.float32)
        vis0[:h, :w] = img  # 填充数据
        # 二维Dct变换
        vis1 = cv2.dct(cv2.dct(vis0))
        vis1.resize(32, 32)
        # 把二维list变成一维list
        img_list = vis1.flatten()
        # 计算均值
        avg = sum(img_list) * 1. / len(img_list)
        avg_list = ['0' if i > avg else '1' for i in img_list]
        # 得到哈希值
        return ''.join(['%x' % int(''.join(avg_list[x:x + 4]), 2) for x in range(0, 32 * 32, 4)])

    # 计算相似度
    def hammingDist(self, s1, s2):
        # 求出相似指数的函数(也看不懂)

        # assert len(s1) == len(s2)
        return 1 - sum([ch1 != ch2 for ch1, ch2 in zip(s1, s2)]) * 1. / (32 * 32 / 4)

    # 打印并返回相似度
    def hash_score(self, ind):
        self.capture_hash(4)
        HASH2 = self.pHash("./photo/garbage_hash.jpg")
        time1 = time.time()
        out_score = self.hammingDist(HASH1, HASH2)
        print('NO.', ind, '|感知哈希算法相似度：', out_score, "-----time=", (time.time() - time1))
        return out_score

    # 计时
    def time_total(self, ind):
        # 用于计算总时间的计时器
        # 大概率以后删除

        global time_start, time_end

        # 参数为1时开始总计时
        if ind == 1:
            time_start = time.time()
            print("开始计时")

        # 参数为0时结束总计时
        if ind == 0:
            time_end = time.time()
            print("结束计时")
            print("总耗时", format(time_end - time_start, '.3f'), "s")  # 保留3位小数


# 单独测试时用的主函数
if __name__ == '__main__':
    SleepTime = 5
    while True:
        DetectMain()
        print("waiting...")
        time.sleep(SleepTime)
