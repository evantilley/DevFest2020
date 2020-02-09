import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import math

# this makes image look better on a macbook pro
def imageshow(img, dpi=200):
    if dpi > 0:
        F = plt.gcf()
        F.set_dpi(dpi)
    plt.imshow(img)

#this is an example
def rgb_ints_example():
    '''should produce red,purple,green squares
    on the diagonal, over a black background'''
    # RGB indexes
    red, green, blue = range(3)
    # img array
    # all zeros = black pixels
    # shape: (150 rows, 150 cols, 3 colors)
    img = np.zeros((150, 150, 3), dtype=np.uint8)
    for x in range(50):
        for y in range(50):
            # red pixels
            img[x, y, red] = 255
            # purple pixels
            # set all 3 color components
            img[x+50, y+50, :] = (128, 0, 128)
            # green pixels
            img[x+100, y+100, green] = 255
    return img

#given an image and a color to separate given as an int where
#red = 0, green = 1, and blue = 2. This image will separate out just that
#color channel from the image
def onechannel(pattern, rgb):
    one_ch_img = np.zeros((pattern.shape[0], pattern.shape[1],\
                           pattern.shape[2]), dtype = pattern.dtype)
    one_ch_img[:,:,rgb] = pattern[:,:,rgb]
    return one_ch_img

#given an image and a list of the color orders to permute to, this function
#will swap the color channels of an image. [0,1,2] is the default
def permutecolorchannels(permcolors, permutation_list):
    permutated_img = np.zeros((permcolors.shape[0], permcolors.shape[1],\
                               permcolors.shape[2]), dtype = permcolors.dtype)
    old_red, old_green, old_blue = range(3)
    new_red, new_green, new_blue = permutation_list
    permutated_img[:, :, old_red] = permcolors[:, :, new_red]
    permutated_img[:, :, old_green] = permcolors[:, :, new_green]
    permutated_img[:, :, old_blue] = permcolors[:, :, new_blue]
    return permutated_img

#decrypt an image given a numpy array for an image and the key file that is
#the XORing key for the image and is as long as the image is by # of cols
def decrypt(image, key):
    decrypted_img = np.zeros((image.shape[0],image.shape[1],image.shape[2]), dtype=image.dtype)
    for col in range(image.shape[1]):
        decrypted_img[:, col, :] = image[:, col, :] ^ key[col]
    return decrypted_img


#load and read the pattern image and display it
pattern = plt.imread('pattern.png')
plt.imshow(pattern)
plt.show()

#display the one channel edited image for the pattern
x = onechannel(pattern, 1)
plt.pause(0.0001)
plt.imshow(x)
plt.show()

#load and read the permcolors image
plt.pause(0.0001)
permcolors = plt.imread('permcolors.jpg')

#permute the permcolors image to correct it and display it
perm = [2,0,1]
plt.pause(0.0001)
plt.imshow(permutecolorchannels(permcolors, perm))
plt.show()

#load and read decrypted image
secret = plt.imread('secret.bmp')
key = np.load('key.npy')

#display decrypted image
plt.pause(0.0001)
plt.imshow(decrypt(secret,key))
plt.show()