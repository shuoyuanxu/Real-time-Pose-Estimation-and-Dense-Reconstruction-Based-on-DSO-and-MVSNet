#!/usr/bin/env python
# license removed for brevity
import os, cv2, time, math
import sys
# sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))


import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from dso_ros.msg import SlidingWindowsMsg
from unimvsnet.msg import DepthMsg

import argparse
from model import Model

import torch
from torchvision import transforms
import numpy as np

#读取一些roslaunch文件的参数
slidingWindowsQueueSize = rospy.get_param("slidingWindowsQueueSize")
depthInfoQueueSize = rospy.get_param("depthInfoQueueSize")
setting_maxFrames = rospy.get_param("setting_maxFrames")

#工具函数，用来将输入的图像resize到限定尺寸内，并调整内参
def scale_mvs_input(img, intrinsics, max_w, max_h, base=32):
    h, w = img.shape[:2]
    if h > max_h or w > max_w:
        scale = 1.0 * max_h / h
        if scale * w > max_w:
            scale = 1.0 * max_w / w
        new_w, new_h = scale * w // base * base, scale * h // base * base  
    else:
        new_w, new_h = 1.0 * w // base * base, 1.0 * h // base * base

    scale_w = 1.0 * new_w / w
    scale_h = 1.0 * new_h / h
    intrinsics[0, :] *= scale_w
    intrinsics[1, :] *= scale_h

    img = cv2.resize(img, (int(new_w), int(new_h)))

    return img, intrinsics

parser = argparse.ArgumentParser(description="UniMVSNet args")


#话题的订阅者和发布者，在全局定义
sub_slidingWindows = []
pub_depth_info = []

#这里是一长串参数解析这里省略掉
#parser.add_argument...

#存放同一滑动窗口内关键帧的列表
Keyframemsg_list = []
#MVSNet的深度假设参数
depth_max = 20
depth_min = 0.01

Windows_id = -1
#收到DSO节点的滑动窗口关键帧信息后，执行的回调函数
def SlidingWindowsCallback(slidingWindowsMsg_input):
    #声明一些全局变量
    global depth_min, depth_max, Windows_id
    #记录输入图像的长宽
    dim = (slidingWindowsMsg_input.image.width,slidingWindowsMsg_input.image.height)
    #step1:获取一组完整的滑动窗口关键帧
    if(slidingWindowsMsg_input.msg_id!=Windows_id):
        Windows_id = slidingWindowsMsg_input.msg_id
        Keyframemsg_list.clear()
        kfmsg = {}
        kfmsg["camToWorld"] = slidingWindowsMsg_input.camToWorld
        kfmsg["Intrinsics"] = slidingWindowsMsg_input.Intrinsics
        kfmsg["image"] = slidingWindowsMsg_input.image
        Keyframemsg_list.append(kfmsg)
        return
    else:
        kfmsg = {}
        kfmsg["camToWorld"] = slidingWindowsMsg_input.camToWorld
        kfmsg["Intrinsics"] = slidingWindowsMsg_input.Intrinsics
        kfmsg["image"] = slidingWindowsMsg_input.image
        Keyframemsg_list.append(kfmsg)
        if len(Keyframemsg_list) < slidingWindowsMsg_input.window_size:
            return
            
    #step2:根据Keyframemsg_list构造一组MVSNet的输入数据
    #包括图像数据imgs，和根据内参和位姿构造的投影矩阵proj_matrices_ms
    #这里的代码基于datasets/dtu_yao.py中的__getitem__函数修改而来
    imgs = []
    proj_matrices = []
    for Keyframemsg in Keyframemsg_list: #for msg
        # read img  
        cv_image = CvBridge().imgmsg_to_cv2(Keyframemsg["image"], "bgr8")
        np_img = np.array(cv_image, dtype=np.float32) / 255.

        # read proj_mat
        extrinsics = np.zeros((4,4))
        for i in range(4): 
            for j in range(4):
                extrinsics[i,j] = Keyframemsg["camToWorld"][4*i+j] 
        intrinsics = np.zeros((3,3))
        intrinsics[0,0] = Keyframemsg["Intrinsics"][0]#fx
        intrinsics[1,1] = Keyframemsg["Intrinsics"][1]#fy
        intrinsics[0,2] = Keyframemsg["Intrinsics"][2]#cx
        intrinsics[1,2] = Keyframemsg["Intrinsics"][3]#cy
        intrinsics[2,2] = 1
        #inverse
        extrinsics = np.linalg.inv(extrinsics)
        #scale?
        intrinsics[:2, :] /= 4.0
        np_img, intrinsics = scale_mvs_input(np_img, intrinsics, args.max_w, args.max_h)
        proj_mat = np.zeros(shape=(2, 4, 4), dtype=np.float32)
        proj_mat[0, :4, :4] = extrinsics
        proj_mat[1, :3, :3] = intrinsics

        imgs.append(np_img)
        proj_matrices.append(proj_mat)
    imgs = np.stack(imgs).transpose([0, 3, 1, 2])
    proj_matrices = np.stack(proj_matrices)
    stage2_pjmats = proj_matrices.copy()
    stage2_pjmats[:, 1, :2, :] = proj_matrices[:, 1, :2, :] * 2
    stage3_pjmats = proj_matrices.copy()
    stage3_pjmats[:, 1, :2, :] = proj_matrices[:, 1, :2, :] * 4
    proj_matrices_ms = {
        "stage1": torch.from_numpy(np.expand_dims(proj_matrices,axis=0)),
        "stage2": torch.from_numpy(np.expand_dims(stage2_pjmats,axis=0)),
        "stage3": torch.from_numpy(np.expand_dims(stage3_pjmats,axis=0))
    }
    
    #step3:使用构造出来的一组数据，运行模型得到outputs
    start_time = time.time()
    outputs = model.test_ros(imgs, proj_matrices_ms, depth_min, depth_max)
    end_time = time.time()
    print('Windows_id {}, Time:{} Res:{}'.format(Windows_id, end_time - start_time, imgs[0].shape))
    
    #step4:根据outputs，构造出话题数据depthmsg并将其发布出去
    #其中深度图和置信图需要作数据格式转换，并resize成原本输入图像的尺寸
    depthmsg = DepthMsg()
    ref_msg = Keyframemsg_list[0]
    depthmsg.image = ref_msg["image"]
    depthmsg.camToWorld = ref_msg["camToWorld"]
    depthmsg.Intrinsics = ref_msg["Intrinsics"]
    Keyframemsg_list.clear()
    depthmsg.depth = CvBridge().cv2_to_imgmsg(cv2.resize(outputs["depth"].transpose(1,2,0),dim,interpolation = cv2.INTER_AREA), encoding='passthrough')
    depthmsg.confidence = CvBridge().cv2_to_imgmsg(cv2.resize(outputs["photometric_confidence"].transpose(1,2,0),dim,interpolation = cv2.INTER_AREA), encoding="passthrough")
    pub_depth_info.publish(depthmsg)
    
if __name__ == '__main__':
    #1.参数解析与模型初始化
    args = parser.parse_args()
    model = Model(args)
    #2.节点初始化，相关话题订阅和发布
    rospy.init_node('unimvsnet_node',anonymous=True)
    sub_slidingWindows = rospy.Subscriber('SlidingWindows',SlidingWindowsMsg,SlidingWindowsCallback,queue_size=slidingWindowsQueueSize)
    pub_depth_info = rospy.Publisher('depth_info',DepthMsg,queue_size=depthInfoQueueSize)

    print("wait for msg")
    rospy.spin()