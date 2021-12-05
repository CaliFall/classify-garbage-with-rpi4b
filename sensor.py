# -*- coding:utf-8 -*-
import time
import RPi.GPIO as GPIO

GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)

import alarm

dist_limit = [00, 20, 10, 20, 20]

def GPIOprepare():
    # GPIO预处理

    # 设置 GPIO 的工作方式 (IN / OUT)
    # 使用BCM编码
    GPIO.setmode(GPIO.BCM)

    # 1(第一象限接口)
    GPIO.setup(20, GPIO.OUT)
    GPIO.setup(21, GPIO.IN)
    # 2(第二象限接口)
    GPIO.setup(22, GPIO.OUT)
    GPIO.setup(23, GPIO.IN)
    # 3(第三象限接口)
    GPIO.setup(24, GPIO.OUT)
    GPIO.setup(25, GPIO.IN)
    # 4(第四象限接口)
    GPIO.setup(26, GPIO.OUT)
    GPIO.setup(27, GPIO.IN)


def distance(channel):
    # 测满的一个函数,下半部分是测距,网上找的

    if channel == 1:
        GPIO_TRIGGER = 20
        GPIO_ECHO = 21
    if channel == 2:
        GPIO_TRIGGER = 22
        GPIO_ECHO = 23
    if channel == 3:
        GPIO_TRIGGER = 24
        GPIO_ECHO = 25
    if channel == 4:
        GPIO_TRIGGER = 26
        GPIO_ECHO = 27

    # 发送高电平信号到 Trig 引脚
    GPIO.output(GPIO_TRIGGER, True)

    # 持续 10 us
    time.sleep(0.00001)
    GPIO.output(GPIO_TRIGGER, False)
    
    start_time = time.time()
    stop_time = time.time()
    kick_time = time.time()
    
    # 记录发送超声波的时刻1
    while GPIO.input(GPIO_ECHO) == 0:
        start_time = float(time.time())
        if time.time() - kick_time >= 0.1:
            return 100

    # 记录接收到返回超声波的时刻2
    while GPIO.input(GPIO_ECHO) == 1:
        stop_time = float(time.time())
        
    # 计算超声波的往返时间 = 时刻2 - 时刻1
    time_elapsed = stop_time - start_time
    # 声波的速度为 343m/s， 转化为 34300cm/s。
    res = (time_elapsed * 34300) / 2

    return res


if __name__ == '__main__':
    GPIO.setmode(GPIO.BCM)
    GPIOprepare()
    
    print("mode=")
    mode = int(input())
 
    if mode == 0:
        while True:
            for channel in range(1,5):
                try:
                    
                    print(str(channel) + ": " + str(distance(channel)))
                    
                    time.sleep(0.05)
                except:
                    print("异常")
            print('===============================')
            time.sleep(0.25)
            
    if mode == 6:
        while True:
            for channel in range(1,5):
                try:
                    dist = distance(channel)
                    print(str(channel) + ": " + str(dist))
                    if dist < dist_limit[channel]:
                         alarm.warningCam()
                         print("NO.",channel," FULL")
                         time.sleep(1)
                    
                    time.sleep(0.05)
                except:
                    print("异常")
            print('===============================')
            time.sleep(0.25)
    
    if mode == 1:
        while 1:
            print("1:",distance(1))
            time.sleep(0.05)
    if mode == 2:
        while 1:
            print("2:",distance(2))
            time.sleep(0.05)
    if mode == 3:
        while 1:
            print("3:",distance(3))
            time.sleep(0.05)
    if mode == 4:
        while 1:
            print("4:",distance(4))
            time.sleep(0.05)
        

