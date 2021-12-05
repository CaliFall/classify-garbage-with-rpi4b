# -*-coding:utf-8 -*-

import RPi.GPIO as GPIO
import time

channel = 12
gap1 = 0.1
gap2 = 0.2
gap_short = 0.001
bbtime = 2
bbtime_repeat = 3

GPIO.setmode(GPIO.BCM)
GPIO.setup(channel, GPIO.OUT)

def warning():
	GPIO.output(channel, False)
	
	for a in range(bbtime_repeat):
		for i in range(bbtime):
			GPIO.output(channel, True)
			time.sleep(gap1)
			GPIO.output(channel, False)
			time.sleep(gap1)
			i += 1
		
		time.sleep(gap2)
		
	GPIO.output(channel, False)
	print("BeepBeep")
	
def warningCam():
	GPIO.output(channel, False)
	GPIO.output(channel, True)
	time.sleep(gap1)
	GPIO.output(channel, False)
	print("BeepBeep")
	
def warningNope():
	GPIO.output(channel, False)
	GPIO.output(channel, True)
	time.sleep(gap_short)
	GPIO.output(channel, False)
	print("BeepBeep")

if __name__ == '__main__':
	warningNope()
