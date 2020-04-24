import cv2
import numpy as np
from PIL import Image, ImageEnhance

###################################################################################

# A utility fucntion to simulate low-light frames for experiments and demo

###################################################################################

def decrease_brightness(input, brightness = 0.03):

    # input = cv2.convertScaleAbs(input)
    # hsv_img = cv2.cvtColor(input, cv2.COLOR_BGR2HSV)
    # h, s, v = cv2.split(hsv_img)
    #
    # v = np.where(v <= brightness, 0, v - brightness)
    # v = np.uint8(v)
    # s = np.where(v <= saturation, 0, v - saturation)
    #
    # #s = np.where(s <= 255 - saturation, s + saturation, 255)
    #
    # edited_hsv_img = cv2.merge((h, s, v))
    # output = cv2.cvtColor(edited_hsv_img, cv2.COLOR_HSV2BGR)

    img = cv2.cvtColor(input, cv2.COLOR_BGR2RGB)
    im_pil = Image.fromarray(img)
    enhancer = ImageEnhance.Brightness(im_pil)
    im_pil = enhancer.enhance(brightness)
    im_np = np.asarray(im_pil)

    # im_np = skimage.util.random_noise(im_np,mode="salt",amount=0.001,seed=1)
    # im_cv = img_as_ubyte(im_np)

    output = cv2.cvtColor(im_np,cv2.COLOR_RGB2BGR)
    return output