import cv2
import numpy as np
from pathlib import Path

def letter(img_path:Path,new_shape:int,limit_big:bool=True):
    img_bgr=cv2.imread(img_path)
    if img_bgr is None:
        raise FileNotFoundError('cant find path')
    # img_rgb=cv2.cvtColor(img_bgr,cv2.COLOR_BGR2RGB)
    h,w=img_bgr.shape[:2]
    scale=max(h/new_shape,w/new_shape)
    if limit_big:
        scale=max(scale,1)
    scale_h,scale_w=int(h/scale),int(w/scale)
    dh,dw=new_shape-scale_h,new_shape-scale_w
    top,left=dh//2,dw//2
    bottom,right=dh-top,dw-left
    ims=cv2.resize(img_bgr,(scale_w,scale_h),interpolation=cv2.INTER_LINEAR)
    ims=cv2.copyMakeBorder(ims,top,bottom,left,right,borderType=cv2.BORDER_CONSTANT,value=114)

    return ims,scale,top,left

if __name__ == '__main__':
    path = Path(r"D:\program\VOCdevkit\VOC2007x\JPEGImages\000018.jpg")
    img,s,a,b=letter(path,640,True)
    cv2.imshow('img',img)
    cv2.waitKey(0)
    print(s)





