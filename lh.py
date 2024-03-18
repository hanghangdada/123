#encoding: UTF-8
#!/usr/bin/env python2
import cv2 as cv
import cv2
import os
import numpy as np
import time
from pymycobot.mycobot import MyCobot
from opencv_yolo import yolo
from VideoCapture import FastVideoCapture
from GrabParams import grabParams
import math
import rospy
from geometry_msgs.msg import Twist
import argparse
import basic
import serial
data=0
ser=serial.Serial("/dev/ttyUSB3",115200,timeout=0.5) #使用USB连接串行口


parser = argparse.ArgumentParser(description='manual to this script')
parser.add_argument("--debug", type=bool, default="True")
args = parser.parse_args()

# height_bias = grabParams.height_bias
# coords = grabParams.coords_ready
# done = grabParams.done





coords_top_ready = [100.3, -63.8, 330, -91.49, 138, -90.53]
coords_top_ready_ok = [100.3, -63.8, 360, -91.49, 138, -90.53]
coords_top_grap = [180.3, -63.8, 360, -91.49, 138, -90.53]
coords_top_grap_ok = [80.3, -63.8, 360, -91.49, 138, -90.53]


class Detect_marker(object):

    def __init__(self):
        super(Detect_marker, self).__init__()

        self.mc = MyCobot(grabParams.usb_dev, grabParams.baudrate)
        self.mc.power_on()
        self.pub = rospy.Publisher('cmd_vel', Twist, queue_size=1)
        rospy.init_node('testpose')
        self.miss_count = 0
        self.rate = rospy.Rate(20) 
        self.yolo = yolo()
        self.aruco_dict = cv.aruco.Dictionary_get(cv.aruco.DICT_6X6_250)
        self.aruco_params = cv.aruco.DetectorParameters_create()
        self.calibrationParams = cv.FileStorage("calibrationFileName.xml", cv.FILE_STORAGE_READ)
        # The coordinates of the grab center point relative to the mycobot
        self.cap = FastVideoCapture(grabParams.cap_num)
        # The coordinates of the cube relative to the mycobot
        self.c_x, self.c_y = grabParams.IMG_SIZE/2, grabParams.IMG_SIZE/2
        # The ratio of pixels to actual values
        self.ratio = grabParams.ratio
        self.dist_coeffs = self.calibrationParams.getNode("distCoeffs").mat()

        height = grabParams.IMG_SIZE
        focal_length = width = grabParams.IMG_SIZE
        self.center = [width / 2, height / 2]
        # Calculate the camera matrix.
        self.camera_matrix = np.array(
            [
                [focal_length, 0, self.center[0]],
                [0, focal_length, self.center[1]],
                [0, 0, 1],
            ],
            dtype=np.float32,
        )
    # Grasping motion

        # 小车后退
    def moveback(self):
        print("backward...")
        count = 25
        move_cmd = Twist()
        while count > 0:
            move_cmd.linear.x = -0.05
            self.pub.publish(move_cmd)
            self.rate.sleep()
            count -= 1

    def moveforward(self):
        print("forward...")
        count = 2
        move_cmd = Twist()
        while count > 0:
            move_cmd.linear.x = 0.05
            self.pub.publish(move_cmd)
            self.rate.sleep()
            count -= 1

    # 小车右转
    def rotate_to_right(self):
        print("rotate_to_right...")
        count = 20
        move_cmd = Twist()
        while count > 0:
            move_cmd.angular.z = -0.05
            self.pub.publish(move_cmd)
            self.rate.sleep()
            count -= 1

    # 小车左转
    def rotate_to_left(self):
        print("rotate_to_left...")
        count = 20
        move_cmd = Twist()
        while count > 0:
            move_cmd.angular.z = 0.05
            self.pub.publish(move_cmd)
            self.rate.sleep()
            count -= 1

    def ready_arm_pose(self):
        basic.grap(False)
        self.mc.send_coords(coords_top_ready,20,0)
        time.sleep(5)


    def move_and_pick(self):
        self.mc.send_coords(coords_top_ready_ok, 20, 0)
        time.sleep(2)
        # self.moveforward()
        self.mc.send_coords(coords_top_grap, 20, 0)
        time.sleep(3)
        basic.grap(True)  # 闭合夹爪

        self.mc.send_coords(coords_top_grap_ok, 20, 0)
        time.sleep(3)


    def place2right(self):

        angles = [-87.27, 0, 0, 0, 0, 0]
        self.mc.send_angles(angles,25)
        time.sleep(3)
        angles = [-87.27, -45.26, 2.28, 1.66, -0.96, 47.02]
        self.mc.send_angles(angles,25)
        time.sleep(3)

        # open
        basic.grap(False)

        angles = [0, 0, 0, 0, 0, 0]
        self.mc.send_angles(angles,25)

    #夹爪闭合
    def pick(self):
        basic.grap(True)

    # 图片缩放，这里默认在yolo类中给出缩放为（640，640）
    def transform_frame(self, frame):
        frame, ratio, (dw, dh) = self.yolo.letterbox(frame, (grabParams.IMG_SIZE, grabParams.IMG_SIZE))

        return frame
   
    
    # detect object
    def obj_detect(self, img):
        self.is_find = False

        # Load ONNX model
        print('--> Loading model')
        net = cv2.dnn.readNetFromONNX(grabParams.ONNX_MODEL)
        # print('done')

        t1 = time.time()
        blob = cv2.dnn.blobFromImage(img, 1 / 255.0, (grabParams.IMG_SIZE, grabParams.IMG_SIZE), [0, 0, 0], swapRB=True, crop=False)
        net.setInput(blob)
        t1 = time.time()
        # Inference
        # print('--> Running model')
        outputs = net.forward(net.getUnconnectedOutLayersNames())[0]

        
        # simple post process
        boxes, classes, scores = self.yolo.yolov5_post_process_simple(outputs)
        t2 = time.time()

        best_result = (0,0,0)
        # img_0 = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        if boxes is not None:
            boxes = boxes*1
            self.yolo.draw(img, boxes, scores, classes)
            for box, score, cl in zip(boxes,scores,classes):
                temp_result = (box, score, cl)
                if score > best_result[1]:
                    best_result = temp_result
                    self.target_image_info = self.get_position_size(box)
                    self.is_find = True
           


        # if self.is_find:
        #     box, score, cl = best_result
        #     self.yolo.draw_single(img, box, score, cl)
        #     self.mc.set_color(255, 0, 0)
        #     print("the object is found")

        self.show_image(img)

    def get_position_size(self,box):
        left,top,right,bottom = box

        x = left + right
        y = bottom + top
        x = x*0.5
        y = y*0.5
        x_size_p = abs(left - right)
        y_size_p = abs(top - bottom)
        return (-(x - self.c_x)), (-(y - self.c_y)), x_size_p, y_size_p

    def rotate_image(self, image, rotate_angle):
        rows,cols  = image.shape[:2]
        angle=rotate_angle
        center = ( cols/2,rows/2)
        heightNew=int(cols*abs(math.sin(math.radians(angle)))+rows*abs(math.cos(math.radians(angle))))
        widthNew=int(rows*abs(math.sin(math.radians(angle)))+cols*abs(math.cos(math.radians(angle))))


        M = cv.getRotationMatrix2D(center,angle,1)
        # print(M)
        M[0,2] +=(widthNew-cols)/2
        M[1,2] +=(heightNew-rows)/2
        # print(M)

        # print(widthNew,heightNew)
        rotated_image  = cv.warpAffine(image,M,(widthNew,heightNew))
        return rotated_image



    def show_image(self, img):
        if grabParams.debug and args.debug:
            cv2.imshow("figure", img)
            cv2.waitKey(50)
        
    def find_target_image(self):
        while not self.is_find:
            img = self.cap.read()
            img = self.rotate_image(img, -90)
            self.obj_detect(img)

    def follow_obj(self):
        self.is_follow_obj_done = False

        while not self.is_follow_obj_done:
            img = self.cap.read()
            img = self.rotate_image(img, -90)
            self.obj_detect(img)
            if self.is_find:
                print("send_cmd_vel")
                self.send_cmd_vel(self.target_image_info)
            else:
                move_cmd = Twist()
                self.pub.publish(Twist())

    def send_cmd_vel(self, info):
        x, y, xsize, ysize = info


        print("chicunzuobiao",info)

        move_cmd = Twist()

        if xsize > 140 or ysize > 140:
            self.pub.publish(Twist())
            self.is_follow_obj_done = True
        elif xsize > 80 or ysize > 80:
            move_cmd.linear.x = 0.03
            if abs(x) > 1:
                move_cmd.angular.z = x/self.cap.getWidth()
                self.pub.publish(move_cmd)
        else:
            move_cmd.linear.x = 0.03
            if abs(x) > 1:
                move_cmd.angular.z = x/self.cap.getWidth()
                self.pub.publish(move_cmd)
                self.rate.sleep()
                print(move_cmd.linear.x, move_cmd.angular.z)

    def run(self):
        while(1):
            global data
            data = ser.read_all()
            #print(data)
        if(data=="b0"):
            self.mc.set_color(0,0,255)#blue, arm is busy
            # self.init_mycobot()
            self.is_find = False
            self.ready_arm_pose()
            self.find_target_image()
            self.follow_obj()
            self.move_and_pick()
            self.place2right()
            cv.destroyAllWindows()


if __name__ == "__main__":

    
    detect = Detect_marker()
    detect.run()   




   
