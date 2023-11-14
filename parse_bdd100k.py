import glob
import os
import shutil
import cv2

class BDD100K:

    def __init__(self,args):
        self.dataset_dir =  args.dataset
        self.save_dir = args.save_dir
        self.im_dir = args.im_dir
        self.dataset_dir = args.data_dir
        self.show_im = args.show_im
        self.split_num = args.split_num
        self.show_imcrop = args.show_imcrop
        self.save_imcrop = args.save_imcrop

    def parse_path(self,path,type="val"):
        file = path.split(os.sep)[-1]
        file_name = file.split(".")[0]
        drivable_file  = file_name + ".png"
        lane_file  = file_name + ".png"
        drivable_path = os.path.join(self.dataset_dir,"labels","drivable","colormaps",type,drivable_file)
        lane_path = os.path.join(self.dataset_dir,"labels","lane","colormaps",type,lane_file)
        return drivable_path,lane_path


    def find_max_value(self,a,b,c):
        max = None
        index = None
        if a>b:
            max=a
            index=1
        else:
            max=b
            index=2
        if c>max:
            max=c
            index=3
        return max,index

    def find_min_value(self,a,b,c):
        min=None
        index=None
        if a<b:
            min=a
            index=1
        else:
            min=b
            index=2
        
        if c<min:
            min=c
            index=3

        return min,index
        

        

    def Get_Vanish_Area(self):
        im_path_list = glob.glob(os.path.join(self.im_dir,"*.jpg"))

        for i in range(len(im_path_list)):
            print(f"{i}:{im_path_list[i]}")
            drivable_path,lane_path = self.parse_path(im_path_list[i])
            print(f"drivable_path:{drivable_path}, \n lane_path:{lane_path}")
            if os.path.exists(drivable_path):
                print("drivable_path exists!")
                drivable_img = cv2.imread(drivable_path)
                if self.show_im:
                    cv2.imshow("drivable",drivable_img)
                    cv2.waitKey(200)
                drivable_h,drivable_w = drivable_img.shape[0],drivable_img.shape[1]
                print(f"drivable_h:{drivable_h},drivable_w:{drivable_w}")
                p1_w =  int(drivable_w / 3.0)
                p2_w =  int(drivable_w / 2.0)
                p3_w =  int( (drivable_w*2.0) / 3.0)
                p1y,p2y,p3y = 0,0,0
                y = 0
                find_small_y = False
                while(y<=drivable_h-1 and find_small_y == False):
                    if(drivable_img[y][p1_w][0]!=0):
                        find_small_y=True
                        p1y = y
                    else:
                        y+=1

                y = 0
                find_small_y = False
                while(y<=drivable_h-1 and find_small_y == False):
                    if(drivable_img[y][p2_w][0]!=0):
                        find_small_y=True
                        p2y = y
                    else:
                        y+=1
                
                y = 0
                find_small_y = False
                while(y<=drivable_h-1 and find_small_y == False):
                    if(drivable_img[y][p3_w][0]!=0):
                        find_small_y=True
                        p3y = y
                    else:
                        y+=1

                print(f"p1y:{p1y},p2y:{p2y},p3y:{p3y}")
                min,index = self.find_min_value(p1y,p2y,p3y)
                print(f"min = {min}, index={index}")

            if os.path.exists(lane_path):
                print("lane_path exists!")
            self.split_Image(im_path_list[i],min,i)

        return min
    
    def split_Image(self,im_path,drivable_min_y,cnt):
        print(im_path)
        label=None
        img = cv2.imread(im_path)
        split_y = int(img.shape[0] / self.split_num)
        h,w = img.shape[0],img.shape[1]
        print(f"split_y:{split_y}")
        for i in range(self.split_num):
            split_img = img[split_y*i:split_y*(i+1),0:w-1]
            
            
            if drivable_min_y>=split_y*i and drivable_min_y<split_y*(i+1):
                print("it is vansih line area")
                label=0
                input()
            else:
                label=1
            
            if self.save_imcrop:
                save_img_dir = os.path.join(self.save_dir,str(label))
                os.makedirs(save_img_dir,exist_ok=True)
                im_name = str(cnt) + "_" + str(i+1) +  ".jpg"
                save_im_path = os.path.join(save_img_dir,im_name)
                cv2.imwrite(save_im_path,split_img)


            if self.show_imcrop:
                cv2.imshow("split img",split_img)
                cv2.waitKey(400)
        return NotImplemented
    

def get_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-imdir','--im-dir',help='image directory',default="/home/ali/Projects/datasets/BDD100K-ori/images/100k/val")
    parser.add_argument('-savedir','--save-dir',help='save image directory',default="/home/ali/Projects/datasets/BDD100K_Val_crop")
    parser.add_argument('-datadir','--data-dir',help='dataset directory',default="/home/ali/Projects/datasets/BDD100K-ori")

    parser.add_argument('-showim','--show-im',type=bool,help='show images',default=False)
    parser.add_argument('-showimcrop','--show-imcrop',type=bool,help='show crop images',default=True)
    parser.add_argument('-saveimcrop','--save-imcrop',type=bool,help='save  crop images',default=True)

    parser.add_argument('-splitnum','--split-num',type=int,help='split number',default=10)
    parser.add_argument('-dataset','--dataset',help='dataset directory',default="/home/ali/Projects/datasets/CULane/driver_161_90frame_crop_2cls/train")
    return parser.parse_args()


if __name__=="__main__":
    args=get_args()
    bk = BDD100K(args)
    bk.Get_Vanish_Area()


