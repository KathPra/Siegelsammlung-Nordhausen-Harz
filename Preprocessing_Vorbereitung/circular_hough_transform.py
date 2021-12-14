### USER INPUT: complete path to image folder
image_folder="/work-ceph/lprasse/siegel/data/train/samples/"

### NO USER INPUT REQUIRED
### Parameters that may be altered: prep_image(resize_factor),image_gradients(sigma, high_threshold, low_threshold),
### hough_trans(min_radius, max_radius, total_num_peaks, threshold), alter_img(buffer)

### Python packages used
import numpy as np
from PIL import Image, ImageDraw
from skimage import data, color, filters, exposure
from skimage.transform import hough_circle, hough_circle_peaks, resize
from skimage.util import img_as_ubyte
from skimage.feature import canny
from scipy import ndimage
import os

### Functions defined
def prep_image(file_name):
    """
    Loads image files, converts them to numpy arrays. As preparation for further steps,
    images are converted to grayscale und shrinked by a factor of 150 in both the x & y dimension.
    For efficiency purposes, the array in converted into a ubyte.
    """
    image_path = os.path.join(image_folder,file_name)
    im = np.asarray(Image.open(image_path))
    gim = color.rgb2gray(im)
    resize_factor = 150
    image_resized = resize(gim, (gim.shape[0] // resize_factor, gim.shape[1] // resize_factor), anti_aliasing=True)
    image = img_as_ubyte(image_resized)
    return(image)

def image_gradients(image):
    """
    The edges of the image are extracted.
    """
    edges = canny(image, sigma=1.5, low_threshold=10, high_threshold=50)
    return edges

def hough_trans(edges, img_out):
    """
    Circular Hough Transformation of the image. For a range of radii, the algorithms detects the top 5 circles in the image.
    Min and max radius are set according to experimental results. 
    Attention: max radius in not included in set of radii. The radii increase by +1 in the interval e.g. [8,9,10,11,12]
    From the top 5 circles, the best circle is returned. When the circle center is too far from the image center, the detected circle is disregarded.
    The threshold in x & y direction is 10 pixel.
    """
    min_radius = 8
    max_radius = 13
    hough_radii = np.arange(min_radius,max_radius,1) # 6,18,1
    hough_res = hough_circle(edges, hough_radii) # extract all possible circle centers for all specified radii
    accums, cx, cy, radii = hough_circle_peaks(hough_res, hough_radii, total_num_peaks=5)
    h,w = img_out.shape # height, width
    for i in range(len(cy)):
        y_max = cy[i] + radii[i]
        y_min = cy[i] - radii[i]
        print(y_min)
        x_max = cx[i] + radii[i]
        x_min = cx[i] - radii[i]
        threshold = 10
        if y_min < threshold or y_max > (h-threshold) or x_max > (w-threshold) or x_min < threshold:
            continue
        cy =cy[i]
        cx = cx[i]
        radii= radii[i]
        break
    return cy,cx,radii

def alter_img(cy, cx, radius):
    """
    The previous step calculated the location of the seal in the image. 
    Now the original image is loaded and converted into an array and cut according the result of the previous step.
    The detected radius is extended by a buffer factor to increase the probability of extracting the entire seal.
    This value is set according to experimental results.
    At the end, the image is cropped to exclusively show the detected seal.
    """
    image_path = os.path.join(image_folder,file_name)
    im = np.asarray(Image.open(image_path))
    h,w,c = im.shape
    # Create same size alpha layer with circle approximation by 20 corner polygon
    alpha = Image.new('L',(w,h),0)
    draw = ImageDraw.Draw(alpha)
    buffer = 1.2
    center_y = int(cy *150)
    center_x = int(cx *150)
    radius = int(radius *150*buffer)
    draw.regular_polygon(bounding_circle=(center_x, center_y, radius), n_sides=30, fill=1)
    # Convert alpha Image to numpy array
    npBackground=np.array(alpha)
    # Add alpha layer to RGB
    dim0 = im[:,:,0]*npBackground
    dim1 = im[:,:,1]*npBackground
    dim2 = im[:,:,2]*npBackground
    result = np.dstack((dim0, dim1, dim2))
    out = Image.fromarray(result).convert("RGB")
    # Cropped image of above dimension
    left = center_x - radius
    top = center_y - radius
    right = center_x + radius
    bottom = center_y + radius
    im1 = out.crop((left, top, right, bottom))
    return im1

def save_imgs(img, file_name):
    """
    The transformed image is saved in various forms:
    high resolution (original size), low resolution (input size required by neural net),
    low resolution and gray scale, low resolution with gray scale and improved contrast, low resolution and edges
    """
    img.save(f'/work-ceph/lprasse/siegel/data/siegel_hq/samples/{file_name}')
    img1 = img.resize((299,299))
    img1.save(f'/work-ceph/lprasse/siegel/data/siegel_lq/samples/{file_name}')
    img2 = img.convert('L')
    img2.save(f'/work-ceph/lprasse/siegel/data/siegel_gray/samples/{file_name}')
    img3 = np.asarray(img2)
    img_adapteq = exposure.equalize_adapthist(img3, clip_limit=0.03)
    out = img_as_ubyte(img_adapteq)
    out1 = Image.fromarray(out.astype(np.uint8), mode='L')
    out1.save(f'/work-ceph/lprasse/siegel/data/siegel_gray_norm/samples/{file_name}')
    edges = canny(out, sigma=1.5, low_threshold=10, high_threshold=50)
    edges = Image.fromarray(edges).convert("RGB")
    edges.save(f'/work-ceph/lprasse/siegel/data/siegel_edges/samples/{file_name}')

### Function call
counter = 0
for file_name in os.listdir(image_folder):
    # progress update
    if counter % 100 == 0:
        print(counter)
    img = prep_image(file_name)
    gradients =image_gradients(img)
    cy, cx, radius = hough_trans(gradients, img)
    out = alter_img(cy, cx, radius)
    save_imgs(out, file_name)
    '''if counter >7500: 
        break'''
    counter +=1