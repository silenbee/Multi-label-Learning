import numpy as np 
from PIL import Image as PIL_Image
from io import StringIO

img_Epoch=5000
for i in range(img_Epoch):
    if i==0:
        continue
    img = PIL_Image.open("E:\\training data\\VG2016\\VG_100K\\" + str(i) + ".jpg")
    img = np.array(img.resize([224, 224]))
    if img.shape!=(224,224,3):
        print(i)
    if i%1000==0:
        print("1000 passed")