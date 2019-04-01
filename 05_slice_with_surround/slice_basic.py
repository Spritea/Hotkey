from PIL import Image
import os

Path="TX/small/p1_edit/"
input="TX/center_gt_edit_final.png"
height=512
width=512
k=0
im = Image.open(input)
imgwidth, imgheight = im.size
for i in range(0,imgheight,height):
    for j in range(0,imgwidth,width):
        box = (j, i, j+width, i+height)
        if j+width<=imgwidth and i+height<=imgheight:
            a = im.crop(box)
        # try:
        #     o = a.crop(area)
        #     o.save(os.path.join(Path,"PNG","%s" % page,"IMG-%s.png" % k))
        # except:
        #     pass
           # k.zfill(5)
            suffix="SAR01-"+"%05d.png"%(633+j/width+(i/height)*39)
            a.save(os.path.join(Path, suffix))
            # a.save(os.path.join(Path, "SAR01-%05d.png" % k))
            k +=1


