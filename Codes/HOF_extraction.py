###### Dense optical flow #########
import cv2
import numpy as np
import glob
import tensorflow as tf
import pandas as pd
# GPU Usage
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES']='2' # choose it from 0 to 7
config = tf.compat.v1.ConfigProto()  # Another Version: config = tf.ConfigProto()
config.gpu_options.allow_growth = True


#video_files = glob.glob('video/subject 10/*left.mp4')
#print(video_files)



video_files = video_files = ['video/subject 10/P002_T010_overground_round_left.mp4']
#print(video_files)


class flow_descriptor():
    def __init__(self, img, bin_size=12):#check bin size later
        self.img = img
        self.bin_size = bin_size
        self.angle_unit = 360 / self.bin_size
        assert type(self.bin_size) == int, "bin_size should be integer,"

    def extract(self):
        mag, ang = cv2.cartToPolar(self.img[:,:,0], self.img[:,:,1],angleInDegrees=True)
        #mag, ang = cv2.cartToPolar(self.img[...,0], self.img[...,1])
        bin_num = ang // self.angle_unit
        
        hist = np.zeros(self.bin_size)
        for i in range(0, self.bin_size):
            hist[i] = mag[bin_num==i].sum()
        hist = hist/(2073600)  #1920*1080
        return hist
#dim = (224,224)
for vid in range(len(video_files)):
    cap = cv2.VideoCapture(video_files[vid])
    length_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  
    #length_total = 10     
    ret, frame1 = cap.read()
    #frame1 = cv2.resize(frame1, dim, interpolation = cv2.INTER_AREA)
    prvs = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
    hsv = np.zeros_like(frame1)
    hsv[...,1] = 255
    
    list_mag = []
    list_ang = []
    count = 0
    flows = []
    while(1):
        ret, frame2 = cap.read()
        #frame2 = cv2.resize(frame2, dim, interpolation = cv2.INTER_AREA)
        next = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)
    
        flow = cv2.calcOpticalFlowFarneback(prvs,next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        flows.append(flow)
        #print(flow.shape)
    
        mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
        count=count+1
        print(video_files[vid], count, length_total)
        list_mag.append(mag)
        list_ang.append(ang)
        hsv[...,0] = ang*180/np.pi/2
        hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
        rgb = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
    
        #cv2.imshow('frame2',rgb)
    #    k = cv2.waitKey(30) & 0xff
        if count == (length_total-1):
            break
        #elif k == ord('s'):
         #   cv2.imwrite('opticalfb.png',frame2)
          #  cv2.imwrite('opticalhsv.png',rgb)
        prvs = next
    
    cap.release()
    #cv2.destroyAllWindows()
    #flows = np.array(flows)

    listHis = []
    for i in range(len(flows)):
        opticalFlow = flow_descriptor(flows[i], bin_size=18)
        hist = opticalFlow.extract()
        listHis.append(hist)
    DF = pd.DataFrame(listHis)    
    save_path = os.path.splitext(video_files[vid])[0]
    DF.to_csv('hof_features/left/'+save_path + '.csv', index=False)

