from pathlib import Path
import natsort
import random
import shutil

Train_Path = Path("I:/DVS_dataset/dataset_final/02_release/train")
Train_File = natsort.natsorted(list(Train_Path.glob("*.bmp")), alg=natsort.PATH)
Train_Str = []
for i in Train_File:
    Train_Str.append(str(i))

Train_Label_Path = Path("I:/DVS_dataset/dataset_final/02_release/train_labels")
Train_Label_File = natsort.natsorted(list(Train_Label_Path.glob("*.bmp")), alg=natsort.PATH)
Train_Label_Str = []
for i in Train_Label_File:
    Train_Label_Str.append(str(i))

random.seed(2017)
train_list=range(len(Train_Str))
val_list=random.sample(train_list,873)
val_list.sort()

val_prefix="I:/DVS_dataset/dataset_final/02_release/val"
val_label_prefix="I:/DVS_dataset/dataset_final/02_release/val_labels"

for k in range(len(val_list)):
    train_name=Train_Str[val_list[k]]
    train_label_name=Train_Label_Str[val_list[k]]
    val_name=val_prefix+'/'+Path(train_name).name
    val_label_name=val_label_prefix+'/'+Path(train_label_name).name
    shutil.move(train_name,val_name)
    shutil.move(train_label_name,val_label_name)

print("kk")