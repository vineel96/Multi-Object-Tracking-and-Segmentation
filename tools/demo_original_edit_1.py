# --------------------------------------------------------
# SiamMask
# Licensed under The MIT License
# Written by Qiang Wang (wangqiang2015 at ia.ac.cn)
# --------------------------------------------------------
import glob,cv2,argparse
from os.path import isfile, join
import numpy as np
import subprocess,os,re,sys
import torch
import randomcolor
import imageio

sys.path.insert(1,'D:\IITH_Research_Intern\MultipleObjectTracking(MOT)\SiamMask-master\experiments\siammask_sharp')
sys.path.insert(1,'D:\IITH_Research_Intern\MultipleObjectTracking(MOT)\SiamMask-master')
from custom import Custom
from tools.test import siamese_init,siamese_track
from utils.config_helper import load_config
from utils.load_helper import load_pretrain

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3"

parser = argparse.ArgumentParser(description='PyTorch Tracking Demo')

parser.add_argument('--resume', default='', type=str, required=True,
                    metavar='PATH',help='path to latest checkpoint (default: none)')
parser.add_argument('--config', dest='config', default='config_davis.json',
                    help='hyper-parameter of SiamMask in json format')
parser.add_argument('--base_path', default='../../data/tennis', help='datasets')
parser.add_argument('--cpu', action='store_true', help='cpu mode')

parser.add_argument('--input_video',type=str,required=True,help='inputvideofile')
parser.add_argument('--classes',type=str,required=True,help='object classes to track in video')

args = parser.parse_args()

#edit yolo by GVA
def get_output_layers(net):
    layer_names = net.getLayerNames()

    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    return output_layers

def draw_prediction(img, class_id, confidence, x, y, x_plus_w, y_plus_h,classes,COLORS):
    label = str(classes[class_id])
    color = COLORS[class_id]
    cv2.rectangle(img, (x,y), (x_plus_w,y_plus_h), color, 2)
    cv2.putText(img, label, (x-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

def detect_object_yolo(image, yolo_available_classes, user_interested_classes):
    Width = image.shape[1]
    Height = image.shape[0]
    scale = 0.00392

    COLORS = np.random.uniform(0, 255, size=(len(yolo_available_classes), 3))
    net = cv2.dnn.readNet("D:\IITH_Research_Intern\MultipleObjectTracking(MOT)\SiamMask-master\\tools\yolov3.weights",
                          "D:\IITH_Research_Intern\MultipleObjectTracking(MOT)\SiamMask-master\\tools\yolov3.cfg")
    blob = cv2.dnn.blobFromImage(image, scale, (416,416), (0,0,0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(get_output_layers(net))
    box_class_ids = []
    confidences = []
    boxes = []
    conf_threshold = 0.5
    nms_threshold = 0.4
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * Width)
                center_y = int(detection[1] * Height)
                w = int(detection[2] * Width)
                h = int(detection[3] * Height)
                x = center_x - w / 2
                y = center_y - h / 2
                box_class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])

    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
    temp_boxes=boxes
    temp_class_ids=box_class_ids
    boxes=[]
    box_class_ids=[]
    for i in indices:
        boxes.append(temp_boxes[i[0]])
        box_class_ids.append(temp_class_ids[i[0]])
    #print("no of objects:",len(boxes))
    #print("classes:",box_class_ids)
    '''for i in indices:
        i = i[0]
        box = temp_boxes[i]
        x = box[0]
        y = box[1]
        w = box[2]
        h = box[3]
        draw_prediction(image, temp_class_ids[i], confidences[i], round(x), round(y), round(x+w), round(y+h),yolo_available_classes,COLORS)
    cv2.imshow("object detection", image)
    cv2.waitKey()
    cv2.destroyAllWindows()
    exit()'''
    #edit by GVA
    yolo_class_id_map=dict((yolo_available_classes[i], i) for i in range(len(yolo_available_classes)))
    user_interested_class_ids=[yolo_class_id_map[name] for name in user_interested_classes]
    for i in user_interested_class_ids:
        if i not in box_class_ids:
           for c,id in yolo_class_id_map.items():
               if i==id:
                  print("Object",c,"is not present in video")
    temp_boxes=[]
    temp_box_class_ids=[]
    for i in range(len(box_class_ids)):
        if box_class_ids[i] in user_interested_class_ids:
            temp_boxes.append(boxes[i])
            temp_box_class_ids.append(box_class_ids[i])
    boxes=temp_boxes
    box_class_ids=temp_box_class_ids
    return boxes,box_class_ids
    #edit end by GVA


def parallel_detect_objects(f,im,state,state_index,device):
    if f > 0:  # tracking
        state[state_index] = siamese_track(state[state_index],state_index, im, mask_enable=True, refine_enable=True, device=device)  # track
        '''location = state[state_index]['ploygon'].flatten()
        mask = state[state_index]['mask'] > state[state_index]['p'].seg_thr

        im[:, :, 2] = (mask > 0) * 255 + (mask == 0) * im[:, :, 2]
        print("frame id:{}, state index: {},object id:{} location coords: {}".format(f,state_index,state[state_index]['track_id'],location))
        print("\n")
        cv2.polylines(im, [np.int0(location).reshape((-1, 1, 2))], True,state[state_index]['track_color'], 3)
        cv2.putText(im,str(state[state_index]['track_id']),(location[2],location[3]),cv2.FONT_HERSHEY_SIMPLEX,0.7,state[state_index]['track_color'],2,cv2.LINE_AA)
        cv2.imshow('SiamMask', im)
        key = cv2.waitKey(1)'''
        #if key > 0:
            #break

if __name__ == '__main__':
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    torch.backends.cudnn.benchmark = True

    # Setup Model
    cfg = load_config(args)
    siammask = Custom(anchors=cfg['anchors'])
    if args.resume:
        assert isfile(args.resume), 'Please download {} first.'.format(args.resume)
        siammask = load_pretrain(siammask, args.resume)

    siammask.eval().to(device)
    # edit by GVA
    data_directory_name=args.input_video[:-4]
    if not os.path.isdir(os.getcwd()+"/"+data_directory_name):
       os.mkdir(os.getcwd()+"/"+data_directory_name)
       subprocess.run(["ffmpeg","-i",args.input_video,"-vf","fps=25",os.getcwd()+"/"+data_directory_name+"/output%06d.jpg"])
    #edit end by GVA

    #edit by GVA
    user_interested_classes=args.classes.split(",")
    for i in range(len(user_interested_classes)):
        if re.findall(r'-', user_interested_classes[i]):
            temp=user_interested_classes[i].split("-")
            user_interested_classes[i]= " ".join(temp)

    with open("D:\IITH_Research_Intern\MultipleObjectTracking(MOT)\SiamMask-master\\tools\yolov3.txt", 'r') as f:
            yolo_available_classes = [line.strip() for line in f.readlines()]

    for i in user_interested_classes:
        if i not in yolo_available_classes:
           print("\n\n\n",i,"is not valid class","\n\n\n")
           print("AVAILABLE CLASSES ARE","\n\n")
           for index in range(len(yolo_available_classes)//11):
               print("  ".join(yolo_available_classes[index*10:(index+1)*10])+"\n")
           exit()

    # Parse Image file
    data_directory_name = 'nuscenes/cam_front_right'
    img_files = sorted(glob.glob(join(os.getcwd()+"/"+data_directory_name, '*.jp*'))) #data_directory_name

    #img_files = sorted(glob.glob(join(args.base_path, '*.jp*')))
    ims = [cv2.imread(imf) for imf in img_files]
    # Select ROI
    '''cv2.namedWindow("SiamMask", cv2.WND_PROP_FULLSCREEN)
    try:
        init_rect = cv2.selectROI('SiamMask', ims[0], False, False)
        x, y, w, h = init_rect
    except:
        exit()'''

    boxes,box_class_ids=detect_object_yolo(ims[0], yolo_available_classes, user_interested_classes)

    toc = 0
    object_colors=[]
    for i in range(1000):
        color = randomcolor.RandomColor().generate()
        hex_val = color[0].lstrip('#')
        rgb = tuple(int(hex_val[j:j + 2], 16) for j in (0, 2, 4))
        object_colors.append(rgb)
    object_ids=[i for i in range(1000)]

    state=[]
    import time
    start_time=time.time()
    track_time=0
    #ims = ims[:120]  #added to get only first 5 seconds, remove it for full video
    print("len of boxes:",len(boxes))
    print("total frames: ",len(ims))
    gif =[]
    #cv2.namedWindow('SiamMask', cv2.WND_PROP_FULLSCREEN)
    for f, im in enumerate(ims):
        tic = cv2.getTickCount()
        if f == 0:
            for i in range(len(boxes)):
                x, y, w, h = boxes[i]
                target_pos = np.array([x + w / 2, y + h / 2])
                target_sz = np.array([w, h])
                color = object_colors.pop()
                id = object_ids.pop()
                state.append({"target_pos":target_pos,"target_sz":target_sz,"track_color":color,"track_id":id})
            init_time=time.time()
            state=siamese_init(im, state, siammask, cfg['hp'],device=device)
            #print("init time:",time.time()-init_time)
        else:
            s=time.time()
            state= siamese_track(f, state, im, mask_enable=True,refine_enable=True, device=device)
            #print("frame, time: ",f,time.time()-s)
            for state_index in range(len(state)):
                location = state[state_index]['ploygon'].flatten()
                mask = state[state_index]['mask'] > state[state_index]['p'].seg_thr
                im[:, :, 2] = (mask > 0) * 255 + (mask == 0) * im[:, :, 2]
                #print("frame id:{}, state index: {},object id:{} location coords: {}".format(f,state_index,state[state_index]['track_id'],location))
                #print("\n")
                location = location.astype('float32')
                print("frame id: ",f)
                print("location type: ",location.dtype, location[2].dtype ,location[3].dtype)
                cv2.polylines(im, [np.int0(location).reshape((-1, 1, 2))], True,state[state_index]['track_color'], 3)
                cv2.putText(im,str(state[state_index]['track_id']),(location[2],location[3]),cv2.FONT_HERSHEY_SIMPLEX,0.7,state[state_index]['track_color'],2,cv2.LINE_AA)
            cv2.imshow('SiamMask', im)
            key = cv2.waitKey(1)
            track_time+=time.time() - s
            gif.append(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
            #print("frame, time: ", f, track_time)

        toc += cv2.getTickCount() - tic

    imageio.mimsave('nuscenes/cam_front_right/cam_front_right.gif',gif,fps=2)
    toc /= cv2.getTickFrequency()
    print("frames:",f," fps:",f/(time.time()-start_time))
    fps = f / toc
    print('SiamMask Time: {:02.1f}s Speed: {:3.1f}fps (with visulization!)'.format(toc, fps))
