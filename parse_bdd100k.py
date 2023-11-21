import glob
import os
import shutil
import cv2

class BDD100K:

    def __init__(self,args):

        ## data directory
        self.dataset_dir =  args.dataset
        self.save_dir = args.save_dir
        self.im_dir = args.im_dir
        self.dataset_dir = args.data_dir
        
        ## split crop image detail
        self.split_num = args.split_num
        self.split_height = args.split_height
        
        self.save_imcrop = args.save_imcrop

        ## Augmentation using shift cropping
        self.multi_crop = args.multi_crop
        self.multi_num = args.multi_num
        self.shift_pixels = args.shift_pixels
        
        ## yolo.txt parameter
        self.save_txtdir = args.save_txtdir
        self.vla_label = args.vla_label
        self.save_img = args.save_img

        ## parse image detail
        self.data_type = args.data_type
        self.data_num = args.data_num
        self.wanted_label_list = [2,3,4]
        # 0: pedestrian
        # 1: rider
        # 2: car
        # 3: truck
        # 4: bus
        # 5: train
        # 6: motorcycle
        # 7: bicycle
        # 8: traffic light
        # 9: traffic sign
        ## view result
        self.show_vanishline = args.show_vanishline
        self.show_imcrop = args.show_imcrop
        self.show_im = args.show_im

    def parse_path(self,path,type="val"):
        file = path.split(os.sep)[-1]
        file_name = file.split(".")[0]
        drivable_file  = file_name + ".png"
        lane_file  = file_name + ".png"
        detection_file = file_name + ".txt"
        drivable_path = os.path.join(self.dataset_dir,"labels","drivable","colormaps",type,drivable_file)
        lane_path = os.path.join(self.dataset_dir,"labels","lane","colormaps",type,lane_file)
        detection_path = os.path.join(self.dataset_dir,"labels","detection",type,detection_file)
        return drivable_path,lane_path,detection_path


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
        

    def Get_Min_y_In_Drivable_Area(self,drivable_path):
        if not os.path.exists(drivable_path):
            drivable_img = cv2.imread(drivable_path)
            return int(drivable_img.shape[0]/2.0)
        else:
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
                    break
                else:
                    y+=1

            y = 0
            find_small_y = False
            while(y<=drivable_h-1 and find_small_y == False):
                if(drivable_img[y][p2_w][0]!=0):
                    find_small_y=True
                    p2y = y
                    break
                else:
                    y+=1
            
            y = 0
            find_small_y = False
            while(y<=drivable_h-1 and find_small_y == False):
                if(drivable_img[y][p3_w][0]!=0):
                    find_small_y=True
                    p3y = y
                    break
                else:
                    y+=1

            print(f"p1y:{p1y},p2y:{p2y},p3y:{p3y}")
            min,index = self.find_min_value(p1y,p2y,p3y)
            if min==0:
                # print(f"min={min} special case~~~~~")
                # if p1y==0 and p2y==0 and p3y==0:
                #     min = int(drivable_img.shape[0]/2.0)
                # elif not all([p1y,p2y,p3y]):
                #     min,index = self.find_max_value(p1y,p2y,p3y)
                min=None
                    
            print(f"min = {min}, index={index}")

            return min,index
        
        
            
    def Find_Min_Y_Among_All_Vehicle_Bounding_Boxes(self,min,detection_path,img_h,img_w):
        print(f"h:{img_h} w:{img_w}")
        min_rea = 999999
        find_min_area=False
        min_x=99999
        min_w=99999
        min_h=99999
        with open(detection_path,"r") as f:
            lines = f.readlines()
            for line in lines:
                find_min_area=False
                #print(line)
                la = line.split(" ")[0]
                x = int(float(line.split(" ")[1])*img_w)
                y = int(float(line.split(" ")[2])*img_h)
                w = int(float(line.split(" ")[3])*img_w)
                h = int(float(line.split(" ")[4])*img_h)
                #print(f"{la} {x} {y} {w} {h}")
                if w*h < min_rea and int(la) in self.wanted_label_list:
                    print(f"w*h={w*h},min_rea={min_rea},x:{x},y:{y}")
                    min_rea = w*h
                    find_min_area=True
                    print(f"find_min_area :{find_min_area} ")
                    
                if min is not None:
                    if int(la) in self.wanted_label_list and find_min_area:
                        print(f"y:{y} min:{min}")
                        min=y
                        min_x=x
                        min_w=w
                        min_h=h
                else:
                    if int(la) in self.wanted_label_list and find_min_area:
                        print(f"y:{y} min:{min}")
                        min=y
                        min_x=x
                        min_w=w
                        min_h=h
        if min is None:
            min = int(img_h/2.0)
        return min
        #return min,min_x,min_w,min_h

    def Get_Vanish_Area(self):
        '''
        func: Get_Vanish_Area
        Purpose : 
            parsing the images in given image directory, 
            find the vanish line area and crop the vanish line area, 
            and crop others area
        input:
            self.im_dir : the image directory
            self.dataset_dir : the dataset directory
            self.save_dir : save crop image directory
            self.multi_crop : save multiple vanish area crop images
        output:
            the split images
        '''
        im_path_list = glob.glob(os.path.join(self.im_dir,"*.jpg"))

        if self.data_num<len(im_path_list):
            final_wanted_img_count = self.data_num
        else:
            final_wanted_img_count = len(im_path_list)

        print(f"final_wanted_img_count = {final_wanted_img_count}")
        min_final_2 = None
        for i in range(final_wanted_img_count):
            print(f"{i}:{im_path_list[i]}")
            drivable_path,lane_path,detection_path = self.parse_path(im_path_list[i],type=self.data_type)
            #print(f"drivable_path:{drivable_path}, \n lane_path:{lane_path}")
            if not os.path.exists(detection_path):
                print(f"detection_path not exist !! PASS:{detection_path}")
                continue
            
            img = cv2.imread(im_path_list[i])
            h,w = img.shape[0],img.shape[1]

            min_final,index = self.Get_Min_y_In_Drivable_Area(drivable_path)
            
            min_final_2 = self.Find_Min_Y_Among_All_Vehicle_Bounding_Boxes(min_final,detection_path,h,w)

            # if os.path.exists(lane_path):
            #     print("lane_path exists!")
            # self.split_Image(im_path_list[i],min_final_2,min_x,min_w,minh)
            self.split_Image(im_path_list[i],min_final_2)


        return min_final_2
    

    def Add_Vanish_Line_Area_Yolo_Txt_Labels(self):
        '''
        func: Get_Vanish_Area
        Purpose : 
            parsing the images in given image directory, 
            find the vanish line area and add bounding box information label x y w h 
            into label.txt of yolo format
        input:
            self.im_dir : the image directory
            self.dataset_dir : the dataset directory
            self.save_dir : save crop image directory
            
        output:
            the label.txt with vanish line area bounding box
        '''
        im_path_list = glob.glob(os.path.join(self.im_dir,"*.jpg"))

        if self.data_num<len(im_path_list):
            final_wanted_img_count = self.data_num
        else:
            final_wanted_img_count = len(im_path_list)

        print(f"final_wanted_img_count = {final_wanted_img_count}")
        min_final_2 = None
        for i in range(final_wanted_img_count):
            print(f"{i}:{im_path_list[i]}")
            drivable_path,lane_path,detection_path = self.parse_path(im_path_list[i],type=self.data_type)
            #print(f"drivable_path:{drivable_path}, \n lane_path:{lane_path}")
            if not os.path.exists(detection_path):
                print(f"detection_path not exist !! PASS:{detection_path}")
                continue
            
            img = cv2.imread(im_path_list[i])
            h,w = img.shape[0],img.shape[1]

            min_final,index = self.Get_Min_y_In_Drivable_Area(drivable_path)
            print(f"min_final = {min_final}")
            #input()
            min_final_2 = self.Find_Min_Y_Among_All_Vehicle_Bounding_Boxes(min_final,detection_path,h,w)

            print(f"min_final_2 :{min_final_2}")
            
            success = self.Add_VLA_Yolo_Txt_Label(min_final_2,detection_path,h,w,im_path_list[i])
            
            #input()
        return min_final_2

    def Add_VLA_Yolo_Txt_Label(self,min_y,detection_path,h,w,im_path):
        success = 0
        # with open(detection_path,'r') as f:
        #     lines = f.readlines()
        #     for line in lines:
        #         print(line)
        txt_file = detection_path.split(os.sep)[-1]
        os.makedirs(self.save_txtdir,exist_ok=True)
        save_txt_path = os.path.join(self.save_txtdir,txt_file)

        ## Copy original label.txt to new directory
        if not os.path.exists(save_txt_path):
            shutil.copy(detection_path,save_txt_path)
            if self.save_img:
                shutil.copy(im_path,self.save_txtdir)
            print(f"Copy detection_path to :{save_txt_path} successful !!")
        else:
            print(f"File {save_txt_path} exists ~~~~~~~~~~~~~~,PASS!!")
            success = 1
            return success

        ## Add new VLA label to label.txt
        VLA_label = self.vla_label
        x = float(int((float(int(w/2.0)-1)/w)*1000000)/1000000)
        y = float(int(float(min_y/h)*1000000)/1000000)
        w = 1.0
        h = float(int(float(self.split_height / h)*1000000)/1000000)
        lxywh =  str(VLA_label) + " "\
                + str(x) + " "\
                + str(y) + " "\
                + str(w) + " "\
                 + str(h)
        with open(save_txt_path,'a') as f:
            # Add VLA(Vanish Line Area Bounding Box) l x y w h
            f.write(lxywh)
        
        success = 1
        return success
    
    # def split_Image(self,im_path,drivable_min_y,min_x,min_w,minh):
    def split_Image(self,im_path,drivable_min_y):
        tag = 0
        retval = 0
        lower_bound_basic = 0
        upper_bound_basic = 0
        if os.path.exists(im_path):
            img_name = (im_path.split(os.sep)[-1]).split(".")[0]
            print(im_path)
            label=None
            y = 0 if drivable_min_y is None else int(drivable_min_y)
            print(f"y:{y}")
            img = cv2.imread(im_path)
            split_y = int(img.shape[0] / self.split_num)
            h,w = img.shape[0],img.shape[1]
            bound_list = []
            if not self.multi_crop:
                lower_bound_basic = y-int(split_y/2.0)
                upper_bound_basic = y+int(split_y/2.0)
                bound_list.append([lower_bound_basic,upper_bound_basic])
            else:
                lower_bound_basic = y-int(split_y/2.0)
                upper_bound_basic = y+int(split_y/2.0)
                bound_list.append([lower_bound_basic,upper_bound_basic])
                for i in range(int(self.multi_num/2.0)):
                    lower_bound = y-int(split_y/2.0)-(i+1)*int(self.shift_pixels)
                    upper_bound = y+int(split_y/2.0)-(i+1)*int(self.shift_pixels)
                    bound_list.append([lower_bound,upper_bound])
                for i in range(int(self.multi_num/2.0)):
                    lower_bound = y-int(split_y/2.0)+(i+1)*int(self.shift_pixels)
                    upper_bound = y+int(split_y/2.0)+(i+1)*int(self.shift_pixels)
                    bound_list.append([lower_bound,upper_bound])

            for i in range(len(bound_list)):
                lower_bound = bound_list[i][0]
                upper_bound = bound_list[i][1]
                print(f"split_y:{split_y}")
                if lower_bound>0:
                    split_vanish_crop_image = img[lower_bound:upper_bound,0:w-1]
                    
                    #input()
                    if self.save_imcrop:
                        label = 0
                        save_dir = os.path.join(self.save_dir,str(label))
                        os.makedirs(save_dir,exist_ok=True)
                        save_crop_im = img_name + "_" + str(tag) + ".jpg"
                        save_crop_im_path = os.path.join(save_dir,save_crop_im)
                        if not os.path.exists(save_crop_im_path):
                            cv2.imwrite(save_crop_im_path,split_vanish_crop_image)
                        else:
                            print(f"file {img_name}_0.jpg exists")
                        tag=tag+1
                    if self.show_imcrop:
                        if self.show_vanishline:
                            newImage = img.copy()
                            cv2.line(newImage, (0, drivable_min_y), (w-1, drivable_min_y), (0, 0, 255), 1)
                            #cv2.rectangle(newImage, start_point, end_point, color, thickness) 
                            split_vanish_crop_image = newImage[lower_bound:upper_bound,0:w-1]
                            cv2.imshow("split vanish area",split_vanish_crop_image)
                            cv2.waitKey(1)
                            input()
                        else:
                            cv2.imshow("split vanish area",split_vanish_crop_image)
                            cv2.waitKey(1)
                else:
                    print("y is too small~~")
                    retval = -1
                    return retval
                
            ## Generate others crop images
            if y>split_y:
                y_l = lower_bound_basic - split_y
                while(y_l>0 and y_l+split_y<h):
                    split_other_image = img[y_l:y_l+split_y,0:w-1]
                    # cv2.imshow("up others img",split_other_image)
                    # cv2.waitKey(100)
                    #input()
                    y_l=y_l-split_y
                    if self.save_imcrop:
                        label = 1
                        save_dir = os.path.join(self.save_dir,str(label))
                        os.makedirs(save_dir,exist_ok=True)
                        save_crop_im = img_name + "_" + str(tag) + ".jpg"
                        save_crop_im_path = os.path.join(save_dir,save_crop_im)
                        if not os.path.exists(save_crop_im_path):
                            cv2.imwrite(save_crop_im_path,split_other_image)
                        else:
                            print(f"file {img_name}_{tag}.jpg exists")
                        tag=tag+1
                y_t = upper_bound_basic + split_y
                while(y_t<(h-1) and y_t-split_y>0):
                    split_other_image = img[y_t-split_y:y_t,0:w-1]
                    # cv2.imshow("down others img",split_other_image)
                    # cv2.waitKey(100)
                    #input()
                    y_t = y_t + split_y
                    if self.save_imcrop:
                        label = 1
                        save_dir = os.path.join(self.save_dir,str(label))
                        os.makedirs(save_dir,exist_ok=True)
                        save_crop_im = img_name + "_" + str(tag) + ".jpg"
                        save_crop_im_path = os.path.join(save_dir,save_crop_im)
                        if not os.path.exists(save_crop_im_path):
                            cv2.imwrite(save_crop_im_path,split_other_image)
                        else:
                            print(f"file {img_name}_{tag}.jpg exists")
                        tag=tag+1
            
        return NotImplemented
    

def get_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-imdir','--im-dir',help='image directory',\
                        default="/home/ali/Projects/datasets/bdd100k_data_0.9/images/100k/val")
    parser.add_argument('-savedir','--save-dir',help='save image directory',\
                        default="/home/ali/Projects/datasets/BDD100K_Val_VLA_label_Txt_h100_2023-11-22")
    parser.add_argument('-datadir','--data-dir',help='dataset directory',\
                        default="/home/ali/Projects/datasets/bdd100k_data_0.9")


    parser.add_argument('-savetxtdir','--save-txtdir',help='save image directory',\
                        default="/home/ali/Projects/datasets/BDD100K_Train_VLA_label_Txt_h100_2023-11-22")
    parser.add_argument('-vlalabel','--vla-label',type=int,help='VLA label',default=12)
    parser.add_argument('-saveimg','--save-img',type=bool,help='save images',default=False)

    parser.add_argument('-datatype','--data-type',help='data type',default="val")
    parser.add_argument('-datanum','--data-num',type=int,help='number of images to crop',default=10000)



    parser.add_argument('-showim','--show-im',type=bool,help='show images',default=False)
    parser.add_argument('-showimcrop','--show-imcrop',type=bool,help='show crop images',default=True)
    parser.add_argument('-showvanishline','--show-vanishline',type=bool,help='show vanish line in image',default=False)
    parser.add_argument('-saveimcrop','--save-imcrop',type=bool,help='save  crop images',default=True)

    parser.add_argument('-multicrop','--multi-crop',type=bool,help='save multiple vanish area crop images',default=True)
    parser.add_argument('-multinum','--multi-num',type=int,help='number of multiple vanish area crop images',default=6)
    parser.add_argument('-shiftpixel','--shift-pixels',type=int,help='number of multiple crop images shift pixels',default=2)

    parser.add_argument('-splitnum','--split-num',type=int,help='split number',default=10)
    parser.add_argument('-splitheight','--split-height',type=int,help='split image height',default=100)
    parser.add_argument('-dataset','--dataset',help='dataset directory',default="/home/ali/Projects/datasets/CULane/driver_161_90frame_crop_2cls/train")
    return parser.parse_args()


if __name__=="__main__":
    args=get_args()
    bk = BDD100K(args)
    #bk.Get_Vanish_Area()
    bk.Add_Vanish_Line_Area_Yolo_Txt_Labels()


