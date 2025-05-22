import cv2
import numpy as np
import pigpio
import threading
import queue
import time

# 硬件初始化
pi = pigpio.pi()
control_queue = queue.Queue(maxsize=1)

# 摄像头参数
CAP_SIZE = (320, 240)
CAP_FPS = 17

# 电机控制引脚
EA, I2, I1, EB, I4, I3, LS, RS = (13, 19, 26, 16, 20, 21, 6, 5)

class PID:

    def __init__(self, P=80, I=0, D=0, speed=0, duty=26):

        self.Kp = P
        self.Ki = I
        self.Kd = D
        self.err_pre = 0
        self.err_last = 0
        self.u = 0
        self.integral = 0
        self.ideal_speed = speed

    def update(self, feedback_value):
        self.err_pre = self.ideal_speed - feedback_value
        self.integral += self.err_pre
        self.u = self.Kp * self.err_pre + self.Ki * self.integral + self.Kd * (self.err_pre - self.err_last)
        self.err_last = self.err_pre
        if self.u > 100:
            self.u = 100
        elif self.u < 0:
            self.u = 0
        return self.u

    def setKp(self, proportional_gain):
        """Determines how aggressively the PID reacts to the current error with setting Proportional Gain"""
        self.Kp = proportional_gain

    def setKi(self, integral_gain):
        """Determines how aggressively the PID reacts to the current error with setting Integral Gain"""
        self.Ki = integral_gain

    def setKd(self, derivative_gain):
        """Determines how aggressively the PID reacts to the current error with setting Derivative Gain"""
        self.Kd = derivative_gain


class PD:

    def __init__(self, P=0, D=0, location=160):  # 120

        self.Kp = P
        self.Kd = D
        self.err_pre = 0
        self.err_last = 0
        self.u = 0
        self.ideal_location = location

    def update(self, feedback_value):
        self.err_pre = feedback_value - self.ideal_location
        self.u = self.Kp * self.err_pre + self.Kd * (self.err_pre - self.err_last)
        self.err_last = self.err_pre
        if self.u > 100:
            self.u = 100
        return self.u

    def setKp(self, proportional_gain):
        """Determines how aggressively the PID reacts to the current error with setting Proportional Gain"""
        self.Kp = proportional_gain

    def setKd(self, derivative_gain):
        """Determines how aggressively the PID reacts to the current error with setting Derivative Gain"""
        self.Kd = derivative_gain

# 全局状态管理
class GlobalState:
    def __init__(self):
        self.lock = threading.Lock()
        self.lspeed = 0
        self.rspeed = 0
        self.lcounter = 0
        self.rcounter = 0
        self.Turn_flag = 0
        self.Red_Eable = 3
        self.Yellow_Eable = 4
        self.Green_Eable = 5
        self.current_target = 'red'  # 当前目标颜色

state = GlobalState()

# 增强型电机控制器
class MotorController:
    def __init__(self):
        self._setup_gpio()
        self.pwma = pi.hardware_PWM(EA, 50, 0)
        self.pwmb = pi.hardware_PWM(EB, 50, 0)

    def _setup_gpio(self):
        pins = [I1, I2, I3, I4]
        for pin in pins:
            pi.set_mode(pin, pigpio.OUTPUT)
        pi.write(I1, 1)
        pi.write(I4, 1)

        pi.set_mode(LS, pigpio.INPUT)
        pi.set_mode(RS, pigpio.INPUT)
        pi.set_pull_up_down(LS, pigpio.PUD_UP)
        pi.set_pull_up_down(RS, pigpio.PUD_UP)

    def set_speed(self, duty_a, duty_b):
        pi.hardware_PWM(EA, 50, int(duty_a * 10000))
        pi.hardware_PWM(EB, 50, int(duty_b * 10000))


# 视觉处理线程（重点修改）
def vision_thread():
    cap = cv2.VideoCapture(0)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAP_SIZE[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAP_SIZE[1])
    cap.set(cv2.CAP_PROP_FPS, CAP_FPS)

    color_config = {
        'red': {
            'lower': np.array([0, 50, 50]),
            'upper': np.array([10, 255, 255]),
            'min_area': 500
        },
        'yellow': {
            'lower': np.array([20, 65, 120]),
            'upper': np.array([40, 230, 230]),
            'min_area': 800
        },
        'green': {
            'lower': np.array([60, 45, 50]),
            'upper': np.array([80, 205, 235]),
            'min_area': 800
        }
    }

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

    while True:
        ret, frame = cap.read()
        if not ret: continue

        # 动态获取当前目标颜色
        with state.lock:
            if (state.Red_Eable == 3):
                current_color = "red"
            elif (state.Yellow_Eable == 3):
                current_color = "yellow"
            elif (state.Green_Eable == 3):
                current_color = "green"

        # 针对当前目标颜色进行处理

        cfg = color_config[current_color]
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, cfg['lower'], cfg['upper'])
        dilated = cv2.dilate(mask, kernel, iterations=1)
        _, contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        target_info = (0, 0, 0, 0)
        if contours:
            max_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(max_contour)
            if w * h > cfg['min_area']:
                target_info = (x, y, w, h)

        control_queue.put((current_color, target_info))


# 控制线程（完整状态机）
def control_thread():
    motor = MotorController()
    L_control = PID(40, 0, 10)
    R_control = PID(40, 0, 11)
    loc_pd = PD(0.002, 0.002, 160)

    TURN_SEQUENCE = {
        'red': (0, 40),  # 左转参数
        'yellow': (40, 0),  # 右转参数
        'green': (20, 40)  # 特殊转向
    }

    while True:
        try:
            color_type, (x, y, w, h) = control_queue.get(timeout=0.1)
            with state.lock:

                if (state.Red_Eable * state.Yellow_Eable * state.Green_Eable == 0):

                    motor.set_speed(0.9)

                elif (color_type == "red"):
                    loc_pd.ideal_location = 160
                    if (h > 180):

                        motor.set_speed(0,0.9)
                        start_time = pi.get_current_tick()
                        while pi.get_current_tick() - start_time < 400000:
                            pass

                        motor.set_speed(0.9,0.9)
                        start_time = pi.get_current_tick()
                        while pi.get_current_tick() - start_time < 400000:
                            pass

                        state.Turn_flag = 1
                        state.Red_Eable = state.Red_Eable - 1
                        state.Yellow_Eable = state.Yellow_Eable - 1
                        state.Green_Eable = state.Green_Eable - 1

                        motor.set_speed(0.9, 0.6)

                    else:

                        center = x + w / 2
                        adjust = loc_pd.update(center)
                        L_control.ideal_speed = 0.9 + adjust
                        R_control.ideal_speed = 0.9 - adjust
                        motor.set_speed(
                            L_control.update(state.lspeed),
                            R_control.update(state.rspeed))

                else:
                    motor.set_speed(0.9, 0.6)
        except queue.Empty:
            continue


# 编码器线程（优化版本）
def encoder_thread():
    def _callback(gpio, level, tick):
        with state.lock:
            if gpio == LS:
                state.lcounter += 1
            elif gpio == RS:
                state.rcounter += 1

    pi.callback(LS, pigpio.RISING_EDGE, _callback)
    pi.callback(RS, pigpio.RISING_EDGE, _callback)

    while True:
        start = pi.get_current_tick()
        while (pi.get_current_tick() - start) < 100000:  # 100ms周期
            pass
        with state.lock:
            state.lspeed = state.lcounter / 585.0
            state.rspeed = state.rcounter / 585.0
            state.lcounter = state.rcounter = 0

threading.Thread(target=vision_thread, daemon=True).start()
threading.Thread(target=encoder_thread, daemon=True).start()
threading.Thread(target=control_thread, daemon=True).start()

try:
    while True:
        continue
except KeyboardInterrupt:
    pass

pi.stop()
cv2.destroyAllWindows()