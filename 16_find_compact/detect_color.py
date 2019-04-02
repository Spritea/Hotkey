from PIL import Image

image = Image.open("E:\code\hotkey\\18_preprocess\Postdam\\train_24_val_14\scale075\\rotate\precode_train_label\scale075_r270-02899.png")
a=image.getcolors()
print(image.getcolors())