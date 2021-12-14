### USER INPUT: complete path to image folder
image_folder="/work-ceph/lprasse/siegel/data/train/samples/"
out_path="/work-ceph/lprasse/siegel/data/test/increase contrast"

### NO USER INPUT REQUIRED
### Parameters that may be altered: save_imgs(clip_limit)

### Python packages used
import numpy as np
from PIL import Image#, ImageDraw
from skimage import exposure
#from skimage import data, color, filters
from skimage.util import img_as_ubyte
#from skimage.feature import canny
#from scipy import ndimage
import os

### Functions defined
def alter_img(file_name):
    """
    The image is opened in a machine readable format and converted into an numpy array.
    """
    image_path = os.path.join(image_folder,file_name)
    im = np.asarray(Image.open(image_path))
    return im

def save_imgs(img, file_name):
    """
    The image is saved improved contrast.
    """
    img_adapteq = exposure.equalize_adapthist(img, clip_limit=0.03)
    out = img_as_ubyte(img_adapteq)
    out1 = Image.fromarray(out.astype(np.uint8))
    out1.save(os.path.join(out_path, file_name))

### Function call
counter = 0
for file_name in os.listdir(image_folder):
    # progress update
    if counter % 100 == 0:
        print(counter)
    out = alter_img(file_name)
    save_imgs(out, file_name)
    counter +=1