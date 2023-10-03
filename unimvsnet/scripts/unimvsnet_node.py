#!/usr/bin/env python3
# license removed for brevity
import os, cv2, time, math
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

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

# Reading paras from roslaunch file
# slidingWindowsQueueSize = rospy.get_param("slidingWindowsQueueSize")
# depthInfoQueueSize = rospy.get_param("depthInfoQueueSize")
# setting_maxFrames = rospy.get_param("setting_maxFrames")
slidingWindowsQueueSize = 10
depthInfoQueueSize = 10
setting_maxFrames = 7

# Resize images and corresponding intrinsic
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

# Definition of subscriber and publisher
sub_slidingWindows = []
pub_depth_info = []

# Parser settings took from main.py
# network
parser.add_argument("--fea_mode", type=str, default="fpn", choices=["fpn", "unet"])
parser.add_argument("--agg_mode", type=str, default="variance", choices=["variance", "adaptive"])
parser.add_argument("--depth_mode", type=str, default="regression",
                    choices=["regression", "classification", "unification"])
parser.add_argument("--ndepths", type=int, nargs='+', default=[48, 32, 8])
parser.add_argument("--interval_ratio", type=float, nargs='+', default=[4, 2, 1])

# dataset
parser.add_argument("--datapath", type=str)
parser.add_argument("--trainlist", type=str)
parser.add_argument("--testlist", type=str)
parser.add_argument("--dataset_name", type=str, default="dtu_yao", choices=["dtu_yao", "general_eval", "blendedmvs"])
parser.add_argument('--batch_size', type=int, default=1, help='train batch size')
parser.add_argument('--numdepth', type=int, default=192, help='the number of depth values')
parser.add_argument('--interval_scale', type=float, default=1.06, help='the number of depth values')
parser.add_argument("--nviews", type=int, default=5)
# only for train and eval
parser.add_argument("--img_size", type=int, nargs='+', default=[512, 640])
parser.add_argument("--inverse_depth", action="store_true")

# training and val
parser.add_argument('--start_epoch', type=int, default=0)
parser.add_argument('--epochs', type=int, default=16, help='number of epochs to train')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--wd', type=float, default=0.0, help='weight decay')
parser.add_argument('--scheduler', type=str, default="steplr", choices=["steplr", "cosinelr"])
parser.add_argument('--warmup', type=float, default=0.2, help='warmup epochs')
parser.add_argument('--milestones', type=float, nargs='+', default=[10, 12, 14], help='lr schedule')
parser.add_argument('--lr_decay', type=float, default=0.5, help='lr decay at every milestone')
parser.add_argument('--resume', type=str, default="/home/shu/catkin_ws/src/unimvsnet/trained_model/unimvsnet_blendedmvs.ckpt", help='path to the resume model')
parser.add_argument('--log_dir', type=str, help='path to the log dir')
parser.add_argument('--dlossw', type=float, nargs='+', default=[0.5, 1.0, 2.0],
                    help='depth loss weight for different stage')
parser.add_argument('--eval_freq', type=int, default=1, help='eval freq')
parser.add_argument('--summary_freq', type=int, default=50, help='print and summary frequency')
parser.add_argument("--val", action="store_true")
parser.add_argument("--sync_bn", action="store_true")
parser.add_argument("--blendedmvs_finetune", action="store_true")

# testing (max_frame modified)
parser.add_argument("--test", action="store_true", default=True)
parser.add_argument('--testpath_single_scene', help='testing data path for single scene')
parser.add_argument('--outdir', default='./outputs', help='output dir')
parser.add_argument('--num_view', type=int, default=setting_maxFrames, help='num of view')
parser.add_argument('--max_h', type=int, default=864, help='testing max h')
parser.add_argument('--max_w', type=int, default=1152, help='testing max w')
parser.add_argument('--fix_res', action='store_true', help='scene all using same res')
parser.add_argument('--num_worker', type=int, default=4, help='depth_filer worker')
parser.add_argument('--save_freq', type=int, default=20, help='save freq of local pcd')
parser.add_argument('--filter_method', type=str, default='gipuma', choices=["gipuma", "pcd", "dypcd"],
                    help="filter method")
parser.add_argument('--display', action='store_true', help='display depth images and masks')
# pcd or dypcd
parser.add_argument('--conf', type=float, nargs='+', default=[0.1, 0.15, 0.9],
                    help='prob confidence, for pcd and dypcd')
parser.add_argument('--thres_view', type=int, default=5, help='threshold of num view, only for pcd')
# dypcd
parser.add_argument('--dist_base', type=float, default=1 / 4)
parser.add_argument('--rel_diff_base', type=float, default=1 / 1300)
# gimupa
parser.add_argument('--fusibile_exe_path', type=str, default='../fusibile/fusibile')
parser.add_argument('--prob_threshold', type=float, default='0.3')
parser.add_argument('--disp_threshold', type=float, default='0.25')
parser.add_argument('--num_consistent', type=float, default='3')

# visualization
parser.add_argument("--vis", action="store_true")
parser.add_argument('--depth_path', type=str)
parser.add_argument('--depth_img_save_dir', type=str, default="./")

# device and distributed
parser.add_argument("--no_cuda", action="store_true")
parser.add_argument("--local_rank", type=int, default=0)
parser.add_argument('--dist-url', default='env://', help='url used to set up distributed training')

args = parser.parse_args()

Keyframemsg_list = []
# Depth assumption for MVSNet
depth_max = 10
depth_min = 0.01
# Windows_id is used to identify if images belongs to the same sliding window
Windows_id = -1
setting_maxFrames = 7


# Callback function for gathering msgs from ros subscriber
def SlidingWindowsCallback(slidingWindowsMsg_input):
    # some global variables
    global depth_min, depth_max, Windows_id
    # obtain the dimension of images
    dim = (slidingWindowsMsg_input.image.width, slidingWindowsMsg_input.image.height)
    # see if images belongs to the current sliding window
    if (slidingWindowsMsg_input.msg_id != Windows_id):  # if not, all Keyframemsg_list needs to be cleared
        Windows_id = slidingWindowsMsg_input.msg_id
        Keyframemsg_list.clear()
        kfmsg = {}
        kfmsg["camToWorld"] = slidingWindowsMsg_input.camToWorld
        kfmsg["Intrinsics"] = slidingWindowsMsg_input.Intrinsics
        kfmsg["image"] = slidingWindowsMsg_input.image
        Keyframemsg_list.append(kfmsg)
        return
    else:  # if id matches, append the image to the Keyframemsg_list
        kfmsg = {}
        kfmsg["camToWorld"] = slidingWindowsMsg_input.camToWorld
        kfmsg["Intrinsics"] = slidingWindowsMsg_input.Intrinsics
        kfmsg["image"] = slidingWindowsMsg_input.image
        Keyframemsg_list.append(kfmsg)
        # loop until Keyframemsg_list reaches the size of the sliding window
        if len(Keyframemsg_list) < slidingWindowsMsg_input.window_size:
            return

    # Formulate the input data for MVSNet using data from Keyframemsg_list
    # including imgs and proj_matrices (camera intrinsic, and extrinsic)
    # partially copied from datasets/dtu_yao.py
    imgs = []
    proj_matrices = []  # Camera intrinsic and extrinsic
    for Keyframemsg in Keyframemsg_list:  # for msg
        # read img (ros img to cv img)
        cv_image = CvBridge().imgmsg_to_cv2(Keyframemsg["image"], "bgr8")
        np_img = np.array(cv_image, dtype=np.float32) / 255.

        # read proj_mat
        extrinsics = np.zeros((4, 4))
        for i in range(4):
            for j in range(4):
                extrinsics[i, j] = Keyframemsg["camToWorld"][4 * i + j]
        intrinsics = np.zeros((3, 3))
        intrinsics[0, 0] = Keyframemsg["Intrinsics"][0]  # fx
        intrinsics[1, 1] = Keyframemsg["Intrinsics"][1]  # fy
        intrinsics[0, 2] = Keyframemsg["Intrinsics"][2]  # cx
        intrinsics[1, 2] = Keyframemsg["Intrinsics"][3]  # cy
        intrinsics[2, 2] = 1
        # inverse
        extrinsics = np.linalg.inv(extrinsics)
        # scale?
        intrinsics[:2, :] /= 4.0
        np_img, intrinsics = scale_mvs_input(np_img, intrinsics, args.max_w, args.max_h)
        proj_mat = np.zeros(shape=(2, 4, 4), dtype=np.float32)
        proj_mat[0, :4, :4] = extrinsics
        proj_mat[1, :3, :3] = intrinsics

        imgs.append(np_img)
        proj_matrices.append(proj_mat)

    # formatting the img and proj_matrices so they can be used as the input of model.py
    imgs = np.stack(imgs).transpose([0, 3, 1, 2])
    proj_matrices = np.stack(proj_matrices)
    stage2_pjmats = proj_matrices.copy()
    stage2_pjmats[:, 1, :2, :] = proj_matrices[:, 1, :2, :] * 2
    stage3_pjmats = proj_matrices.copy()
    stage3_pjmats[:, 1, :2, :] = proj_matrices[:, 1, :2, :] * 4
    proj_matrices_ms = {
        "stage1": torch.from_numpy(np.expand_dims(proj_matrices, axis=0)),
        "stage2": torch.from_numpy(np.expand_dims(stage2_pjmats, axis=0)),
        "stage3": torch.from_numpy(np.expand_dims(stage3_pjmats, axis=0))
    }

    # run test_ros to obtain outputs
    start_time = time.time()
    outputs = model.test_ros(imgs, proj_matrices_ms, depth_min, depth_max)
    end_time = time.time()
    print('Windows_id {}, Time:{} Res:{}'.format(Windows_id, end_time - start_time, imgs[0].shape))
    print('probability:',  np.mean(outputs["photometric_confidence"]))
    # formulate DepthMsg using output then publish it
    # depth map and confidence map needs to be converted to ros format and resize to the original image size
    depthmsg = DepthMsg()
    ref_msg = Keyframemsg_list[0]
    depthmsg.image = ref_msg["image"]
    depthmsg.camToWorld = ref_msg["camToWorld"]
    depthmsg.Intrinsics = ref_msg["Intrinsics"]
    Keyframemsg_list.clear()
    depthmsg.depth = CvBridge().cv2_to_imgmsg(
        cv2.resize(outputs["depth"].transpose(1, 2, 0), dim, interpolation=cv2.INTER_AREA), encoding='passthrough')
    depthmsg.confidence = CvBridge().cv2_to_imgmsg(
        cv2.resize(outputs["photometric_confidence"].transpose(1, 2, 0), dim, interpolation=cv2.INTER_AREA),
        encoding="passthrough")
    pub_depth_info.publish(depthmsg)


if __name__ == '__main__':
    # Parsing settings and initialising model
    args = parser.parse_args()
    model = Model(args)
    # node initialisation, subscriber, and publisher definition
    rospy.init_node('unimvsnet_node', anonymous=True)
    sub_slidingWindows = rospy.Subscriber('SlidingWindows', SlidingWindowsMsg, SlidingWindowsCallback,
                                          queue_size=slidingWindowsQueueSize)
    pub_depth_info = rospy.Publisher('depth_info', DepthMsg, queue_size=depthInfoQueueSize)
    print("wait for msg")
    rospy.spin()
