import glob
import os
import shutil
import cv2

class CULane:
    def __init__(self,args):
        self.im_dir = args.im_dir
        self.save_dir = args.save_dir
        self.show_im = args.show_im
        self.show_imcrop = args.show_imcrop
        self.save_imcrop = args.save_imcrop
        self.dataset = args.dataset
        self.labels = [0,1]

    def Balance_Data(self):
        label_num = []
        for label in self.labels:
            label_path_list = glob.glob(os.path.join(self.dataset,str(label),"*.jpg"))
            label_count = len(label_path_list)
            label_num.append(label_count)
            
        # print(f"min(label_num):{min(label_num)}")
        # print(f"max(label_num):{max(label_num)}")
        # ratio = max(label_num) / min(label_num)
        for i in range(len(label_num)): # search index label
            print(f"{i}:{label_num[i]}")
            ratio = max(label_num) / label_num[i] 
            if int(ratio) > 1:
                im_path_list = glob.glob(os.path.join(self.dataset,str(i),"*.jpg"))
                for j in range(len(im_path_list)):
                    for k in range(int(ratio-1)):
                        im_path  = im_path_list[j]
                        print(f"{k} : im_path:{im_path}")
                        im_name = (im_path.split(os.sep)[-1]).split(".")[0]
                        copy_im_file = im_name + "_" + str(k+2) + ".jpg"
                        copy_im_path = os.path.join(self.dataset,str(i),copy_im_file)
                        shutil.copy(im_path,copy_im_path)

    def Split_Images(self,split_num=10):
        im_path_list = glob.glob(os.path.join(self.im_dir,"***","**","*.jpg"))
        if len(im_path_list)==0:
            im_path_list = glob.glob(os.path.join(self.im_dir,"**","*.jpg"))
        if len(im_path_list)==0:
            im_path_list = glob.glob(os.path.join(self.im_dir,"*.jpg"))

        im_cnt = 1
        for i in range(len(im_path_list)):
            print(f"{i}:{im_path_list[i]}")
            im = cv2.imread(im_path_list[i])

            w,h = im.shape[1], im.shape[0]
            print(f"w:{w},h:{h}")

            if self.show_im:
                cv2.imshow("culane_im",im)
                cv2.waitKey(100)

            h_split = h/split_num
            print(f"h_split = {h_split}")
            for i in range(split_num):
                print(f"h_split*i = {h_split*i}")
                crop_im = im[int(h_split*i):int(h_split*(i+1)-1) , 0:(w-1)]
                
                if self.show_imcrop:
                    cv2.imshow("im_crop",crop_im)             
                    cv2.waitKey(500)
                if self.save_imcrop:
                    os.makedirs(self.save_dir,exist_ok=True)
                    save_dir = os.path.join(self.save_dir,str(i))
                    os.makedirs(save_dir,exist_ok=True)
                    crop_im_file = str(im_cnt) + ".jpg"
                    save_cropim_path = os.path.join(save_dir,crop_im_file)
                    if not os.path.exists(save_cropim_path):
                        cv2.imwrite(save_cropim_path,crop_im)
                    else:
                        print(f"crop im {crop_im_file} already exists, PASS !")
                    print(f"save crop im {im_cnt}.jpg successful")

                im_cnt+=1


def get_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-imdir','--im-dir',help='image directory',default="/home/ali/Projects/datasets/CULane/driver_161_90frame")
    parser.add_argument('-savedir','--save-dir',help='save image directory',default="/home/ali/Projects/datasets/CULane/driver_161_90frame_crop")

    parser.add_argument('-showim','--show-im',type=bool,help='show images',default=False)
    parser.add_argument('-showimcrop','--show-imcrop',type=bool,help='show crop images',default=False)
    parser.add_argument('-saveimcrop','--save-imcrop',type=bool,help='save  crop images',default=True)


    parser.add_argument('-dataset','--dataset',help='dataset directory',default="/home/ali/Projects/datasets/CULane/driver_161_90frame_crop_2cls")
    return parser.parse_args()


if __name__=="__main__":
    args = get_args()
    cu = CULane(args)
    #cu.Split_Images()
    cu.Balance_Data()
    