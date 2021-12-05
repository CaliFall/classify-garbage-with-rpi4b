
# -*- coding: utf-8 -*-
import time
import Adafruit_PCA9685
try:
    pwm = Adafruit_PCA9685.PCA9685()
    pwm.set_pwm_freq(50)
    
    channel_lower = 0
    channel_upper = 4
    biggap1 = 0.50
    gap1 = 0.10
    gap2 = 0.40
    flip_angle = 60     # 上舵机翻转角度
    offset_l = 2        # 下舵机修正角度
    offset_u = -9       # 上舵机修正角度
    
    moto_enable = True
    ddmode = False
except:
    print("舵机初始化失败")
    pass

def set_servo_angle(channel, angle):
    # 输入角度转换成12^精度的数值
    # + 0.5是进行四舍五入运算。
    date = int(4096 * ((angle * 11) + 500) / 20000 + 0.5)
    pwm.set_pwm(channel, 0, date)

def motoact(mode):
    if not moto_enable:
        return 0

    if mode == 1:  # 第一象限
        set_servo_angle(channel_lower, (90 - 45 / 2) + offset_l)
        time.sleep(gap1)
        set_servo_angle(channel_upper, 90 + offset_u + flip_angle)
        time.sleep(gap2)
        motoact(0)
    if mode == 2:  # 第二象限
        set_servo_angle(channel_lower, (90 + 45 / 2) + offset_l)
        time.sleep(gap1)
        set_servo_angle(channel_upper, 90 + offset_u + flip_angle)
        time.sleep(gap2)
        motoact(0)
    if mode == 3:  # 第三象限
        set_servo_angle(channel_lower, (90 - 45 / 2) + offset_l)
        time.sleep(gap1)
        set_servo_angle(channel_upper, 90 + offset_u - flip_angle)
        time.sleep(gap2)
        motoact(0)
    if mode == 4:  # 第四象限
        set_servo_angle(channel_lower, (90 + 45 / 2) + offset_l)
        time.sleep(gap1)
        set_servo_angle(channel_upper, 90 + offset_u - flip_angle)
        time.sleep(gap2)
        motoact(0)
        
    if mode == 0:  # 复位动作
        set_servo_angle(channel_upper, 90 + offset_u)
        set_servo_angle(channel_lower, 90 + offset_l)
        
    if mode == 5:  # 测试动作
        motoact(1)
        time.sleep(0.5)
        motoact(2)
        time.sleep(0.5)
        motoact(3)
        time.sleep(0.5)
        motoact(4)
        time.sleep(0.5)
        
    if mode == 6:  # 调试动作
        set_servo_angle(channel_lower, 90 - 45 / 2 + offset_l)
        set_servo_angle(channel_upper, 165 + offset_u)

    if mode == 7:  # 警告动作
        set_servo_angle(channel_lower, 90 + offset_u + 20)
        time.sleep(0.5)
        set_servo_angle(channel_lower, 90 + offset_u - 20)
        time.sleep(0.5)
        motoact(0)

    if mode == 8:  # 调头动作
        set_servo_angle(channel_lower, (90 - 180 /2) + offset_u - 2)

def motoact_dd(mode, ddflag):
    if not moto_enable:
        return 0

    if mode == 1:  # 第一象限
        if ddflag == "A":
            set_servo_angle(channel_lower, (90 - 45 / 2) + offset_l)
            time.sleep(gap1)
            set_servo_angle(channel_upper, 90 + offset_u + flip_angle)
            time.sleep(gap2)
            motoact(0)
        if ddflag == "B":
            set_servo_angle(channel_lower, (90 + 135 / 2) + offset_l)
            time.sleep(biggap1)
            set_servo_angle(channel_upper, 90 + offset_u - flip_angle)
            time.sleep(gap2)
            motoact(0)

    if mode == 2:  # 第二象限
        if ddflag == "A":
            set_servo_angle(channel_lower, (90 + 45 / 2) + offset_l)
            time.sleep(gap1)
            set_servo_angle(channel_upper, 90 + offset_u + flip_angle)
            time.sleep(gap2)
            motoact(0)
        if ddflag == "B":
            set_servo_angle(channel_lower, (90 - 135 / 2) + offset_l)
            time.sleep(biggap1)
            set_servo_angle(channel_upper, 90 + offset_u - flip_angle)
            time.sleep(gap2)
            motoact(0)

    if mode == 3:  # 第三象限
        if ddflag == "A":
            set_servo_angle(channel_lower, (90 + 135 / 2) + offset_l)
            time.sleep(biggap1)
            set_servo_angle(channel_upper, 90 + offset_u + flip_angle)
            time.sleep(gap2)
            motoact(0)
        if ddflag == "B":
            set_servo_angle(channel_lower, (90 - 45 / 2) + offset_l)
            time.sleep(gap1)
            set_servo_angle(channel_upper, 90 + offset_u - flip_angle)
            time.sleep(gap2)
            motoact(0)

    if mode == 4:  # 第四象限
        if ddflag == "A":
            set_servo_angle(channel_lower, (90 - 135 / 2) + offset_l)
            time.sleep(biggap1)
            set_servo_angle(channel_upper, 90 + offset_u + flip_angle)
            time.sleep(gap2)
            motoact(0)
        if ddflag == "B":
            set_servo_angle(channel_lower, (90 + 45 / 2) + offset_l)
            time.sleep(gap1)
            set_servo_angle(channel_upper, 90 + offset_u - flip_angle)
            time.sleep(gap2)
            motoact(0)

    if mode == 0:  # 复位动作
        set_servo_angle(channel_upper, 90 + offset_u)
        set_servo_angle(channel_lower, 90 + offset_l)

    if mode == 5:  # 测试动作
        motoact(1)
        time.sleep(0.5)
        motoact(2)
        time.sleep(0.5)
        motoact(3)
        time.sleep(0.5)
        motoact(4)
        time.sleep(0.5)

    if mode == 6:  # 调试动作
        set_servo_angle(channel_lower, 90 - 45 / 2 + offset_l)
        set_servo_angle(channel_upper, 165 + offset_u)

    if mode == 7:  # 警告动作
        set_servo_angle(channel_lower, 90 + offset_u + 20)
        time.sleep(0.5)
        set_servo_angle(channel_lower, 90 + offset_u - 20)
        time.sleep(0.5)
        motoact(0)

    if mode == 8:  # 调头动作
        set_servo_angle(channel_lower, (90 - 180 /2) + offset_u - 2)


if __name__ == '__main__':
    mode = 1
    
    if mode == 1:
        while True:
            key = int(input("key:"))
            if ddmode:
                motoact_dd(key, ddflag)
            else:
                motoact(key)

    if mode == 2:
        while True:
            motoact(1)
            time.sleep(0.5)
            motoact(2)
            time.sleep(0.5)
            motoact(3)
            time.sleep(0.5)
            motoact(4)
            time.sleep(0.5)
            
    if mode == 3:
        while True:
            set_servo_angle(4, int(input()))
