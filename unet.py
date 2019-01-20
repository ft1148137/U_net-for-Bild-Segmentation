import os
import sys
import numpy as np
import tensorflow as tf
import random
import math
import warnings
import pandas as pd
import cv2
import matplotlib.pyplot as plt


from PIL import Image

from tqdm import tqdm
from itertools import chain
from skimage.io import imread, imshow, imread_collection, concatenate_images
from skimage.transform import resize
from skimage.morphology import label
np.set_printoptions(threshold='nan')
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def random_crop(image, mask, crop_shape, padding=None):
    mask = resize(mask, (IMG_HEIGHT, IMG_WIDTH,1), mode='constant',preserve_range=True)
    image = resize(image, (IMG_HEIGHT, IMG_WIDTH,IMG_CHANNELS), mode='constant',preserve_range=True)
    img_h = image.shape[0]
    img_w = image.shape[1]
    img_d = image.shape[2]
    mask_d = 1
    if padding:
        oshape_h = img_h + 2 * padding
        oshape_w = img_w + 2 * padding
        img_pad = np.zeros([oshape_h, oshape_w, img_d], np.uint8)
        mask_pad = np.zeros([oshape_h, oshape_w, mask_d], np.uint8)
        img_pad[padding:padding+img_h, padding:padding+img_w, 0:img_d] = image
        mask_pad[padding:padding+img_h, padding:padding+img_w, 0:mask_d] = mask
  
        nh = random.randint(0, oshape_h - crop_shape[0])
        nw = random.randint(0, oshape_w - crop_shape[1])
        image_crop = img_pad[nh:nh + crop_shape[0], nw:nw + crop_shape[1]]
        mask_crop = mask_pad[nh:nh + crop_shape[0], nw:nw + crop_shape[1]]
        return image_crop,mask_crop
    else:
        print("WARNING!!! nothing to do!!!")
        return image,mask

def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def upconv_concat(inputA, input_B, n_filter, name_):
    up_conv = upconv_2D(inputA, n_filter, name_)

    return tf.concat([up_conv, input_B], axis=3, name= name_)


def upconv_2D(tensor, n_filter, name_):
    return tf.layers.conv2d_transpose(
        tensor,
        filters=n_filter,
        kernel_size=2,
        strides=2,
	    activation=tf.nn.relu,
        use_bias=True,
        bias_initializer=tf.zeros_initializer(),
        ##kernel_regularizer=tf.contrib.layers.l2_regularizer(0.01),
        name = name_)

def randomRotation(image, mask,mode=Image.BICUBIC):
    im = resize(image, (IMG_HEIGHT, IMG_WIDTH,IMG_CHANNELS), mode='constant', preserve_range=True)  
    mk = resize(mask, (IMG_HEIGHT, IMG_WIDTH,1), mode='constant', preserve_range=True) 
    random_angle = np.random.randint(1, 360)
    rows,cols = img.shape[:2]
    M = cv2.getRotationMatrix2D((cols/2,rows/2),random_angle,1)
    im = cv2.warpAffine(im.astype(np.uint8),M,(cols,rows))
    mk =  cv2.warpAffine(mk.astype(np.uint8),M,(cols,rows))
    return im, mk

def conv2d(input_tensor, depth, kernel, name, strides=(1, 1), padding="SAME"):
    return tf.layers.conv2d(input_tensor, filters=depth, kernel_size=[kernel,kernel],
                            strides=strides, padding=padding,## kernel_regularizer=tf.contrib.layers.l2_regularizer(0.01),
                            activation=tf.nn.relu,
                            use_bias=True,
                            bias_initializer=tf.zeros_initializer())

def shuffle():
    global image_mix, label_mix
    p = np.random.permutation(len(X_train))
    image_mix = X_train[p]
    label_mix = Y_train[p]
    

def next_batch(batch_s, iters):
    if(iters == 0):
        shuffle()
    count = batch_s * iters
    return image_mix[count:(count + batch_s)], label_mix[count:(count + batch_s)]

warnings.filterwarnings('ignore', category=UserWarning, module='skimage')


IMG_WIDTH = 512
IMG_HEIGHT = 384
IMG_CHANNELS = 1
TRAIN_PATH = '/home/shaoxuan/tensorflow_ws/script/data/train'
TEST_PATH = '/home/shaoxuan/tensorflow_ws/script/data/test'


train_ids = os.listdir(TRAIN_PATH + "/images") 
train_mask = os.listdir(TRAIN_PATH + "/masks") 
test_ids = os.listdir(TEST_PATH + "/images") 
test_mask = os.listdir(TEST_PATH + "/masks")


images = np.zeros((len(train_ids*3), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
labels = np.zeros((len(train_ids*3), IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)
images_t = np.zeros((len(test_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
labels_t = np.zeros((len(test_ids), IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)

print('Getting and resizing train images and masks ... ')
sys.stdout.flush()
mask = np.zeros((IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)
for n, id_ in tqdm(enumerate(train_ids), total=len(train_ids)):
    img = cv2.imread(TRAIN_PATH + "/images/" + id_)
    mask_ = cv2.imread(TRAIN_PATH + '/masks/' + id_)
    mask_ = cv2.cvtColor(mask_,cv2.COLOR_BGR2GRAY)
    fake_img_1, fake_mask_1 = randomRotation(img,mask_)
    fake_img_2, fake_mask_2 = random_crop(img,mask_,[IMG_HEIGHT*7/8,IMG_WIDTH*7/8],True)
    img = resize(img, (IMG_HEIGHT, IMG_WIDTH,IMG_CHANNELS), mode='constant', preserve_range=True)
    mask_ = np.expand_dims(resize(mask_, (IMG_HEIGHT, IMG_WIDTH), mode='constant',preserve_range=True), axis=-1)
    fake_img_1 = resize(fake_img_1, (IMG_HEIGHT, IMG_WIDTH,IMG_CHANNELS), mode='constant', preserve_range=True)
    fake_mask_1 = np.expand_dims(resize(fake_mask_1, (IMG_HEIGHT, IMG_WIDTH), mode='constant',preserve_range=True), axis=-1)
    fake_img_2 = resize(fake_img_2, (IMG_HEIGHT, IMG_WIDTH,IMG_CHANNELS), mode='constant', preserve_range=True)
    fake_mask_2 = resize(fake_mask_2, (IMG_HEIGHT, IMG_WIDTH,1), mode='constant',preserve_range=True)
    images[n] = img
    labels[n] = mask_
    images[n+len(train_ids)] = fake_img_1
    labels[n+len(train_ids)] = fake_mask_1
    images[n+2*len(train_ids)] = fake_img_2
    labels[n+2*len(train_ids)] = fake_mask_2

X_train = images
Y_train = labels 
print X_train.shape

for n, id_ in tqdm(enumerate(test_ids), total=len(test_ids)):
    img = cv2.imread(TEST_PATH + "/images/" + id_)
    img = resize(img, (IMG_HEIGHT, IMG_WIDTH,IMG_CHANNELS), mode='constant', preserve_range=True)
    images_t[n] = img
    mask_ = cv2.imread(TEST_PATH + '/masks/' + id_)
    mask_ = cv2.cvtColor(mask_,cv2.COLOR_BGR2GRAY)
    mask_ = resize(mask_, (IMG_HEIGHT, IMG_WIDTH,1), mode='constant', preserve_range=True)
    labels_t[n] = mask_

X_test = images_t
Y_test = labels_t



X = tf.placeholder(tf.float32, [None, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS])
Y_ = tf.placeholder(tf.bool, [None, IMG_HEIGHT, IMG_WIDTH, 1])
lr = tf.placeholder(tf.float32)
keep_prob = tf.placeholder(tf.float32)
##X = tf.contrib.layers.bias_add(X,initializer=tf.zeros_initializer())
X = X/255
conv1 = conv2d(X, 32, 3,"conv1.1") 
conv1 = conv2d(conv1, 32, 3,"conv1.2") 
 

pool1 = tf.layers.max_pooling2d(conv1, (2, 2),strides=(2, 2), name="pool1")
print pool1
conv2 = tf.contrib.layers.bias_add(pool1,initializer=tf.zeros_initializer())
conv2 = conv2d(pool1, 64, 3, "conv2.1")
conv2 = conv2d(conv2, 64, 3, "conv2.2")


pool2 = tf.layers.max_pooling2d(conv2, (2, 2),strides=(2, 2), name="pool2")
print pool2
conv3 = tf.contrib.layers.bias_add(pool2,initializer=tf.zeros_initializer())
conv3 = conv2d(pool2, 128, 3, "conv3.1")
conv3 = conv2d(conv3, 128, 3, "conv3.2")


pool3 = tf.layers.max_pooling2d(conv3, (2, 2), strides=(2, 2), name="pool3") 
print pool3
conv4 = tf.contrib.layers.bias_add(pool3,initializer=tf.zeros_initializer())
conv4 = conv2d(pool3, 256, 3, "conv4.1")
conv4 = conv2d(conv4, 256, 3, "conv4.2")

pool4 = tf.layers.max_pooling2d(conv4, (2, 2), strides=(2, 2), name="pool4") 
print pool4
conv5 = tf.contrib.layers.bias_add(pool4,initializer=tf.zeros_initializer())
conv5 = conv2d(pool4, 512, 3, "conv5.1")
conv5 = conv2d(conv5, 512, 3, "conv5.2")
 
up6 = upconv_concat(conv5,conv4,64, "up6")
##up6 = tf.nn.dropout(up6,keep_prob)
conv6 = conv2d(up6, 256, 3, "conv6.1") 
conv6 = conv2d(conv6, 256, 3, "conv6.2")
print up6
up7 = upconv_concat(conv6,conv3,32, "up7")
##up7 = tf.nn.dropout(up7,keep_prob)
conv7 = conv2d(up7, 128, 3, "conv7.1")
conv7 = conv2d(conv7, 128, 3, "conv7.2")
print up7
up8 = upconv_concat(conv7,conv2,16, "up8")
##up8 = tf.nn.dropout(up8,keep_prob)
conv8 = conv2d(up8, 64, 3, "conv8.1") 
conv8 = conv2d(conv8, 64, 3, "conv8.2")
print up8
up9 = upconv_concat(conv8,conv1,8, "up9")
##up9 = tf.nn.dropout(up9,keep_prob)
conv9 = conv2d(up9, 32, 3, "conv9.1") 
conv9 = conv2d(conv9, 32, 3, "conv9.2")
print up9
logit = tf.layers.conv2d(conv9, 1, (1, 1),name='final',padding='same')
logit = logit 
print "logit: ",logit



################################
##Y_ = tf.cast(Y_,dtype = tf.float32)
##logit = tf.nn.softmax(logit)

##logit = tf.nn.sigmoid(logit)
##loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits_v2(labels = Y_,logits = logit))
##loss = IOU_(logit,Y_)
loss = tf.losses.sigmoid_cross_entropy(Y_, logit)
##loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=Y_,logits=logit))
##print loss
##optimizer = tf.train.AdamOptimizer(lr).minimize(loss)
##optimizer = tf.train.MomentumOptimizer(lr,0.9).minimize(loss)
optimizer = tf.train.RMSPropOptimizer(lr).minimize(loss)

# init
init = tf.global_variables_initializer()
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction = 0.99)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

sess.run(init)
writer = tf.summary.FileWriter('./logs/train', sess.graph)

saver = tf.train.Saver()

batch_count = 0
display_count = 1
rate = 0.0001
time = 0
num_epochs = 15
batch_size = 3
train_loss_results = []
train_epoch_results = []
val_epoch_results = []
iterations = num_epochs * len(X_train) // batch_size

for i in range(iterations):



##################
##    if time == 3:  
##        test_image = X_test[3]
##        test_mask = np.reshape(test_image, [IMG_HEIGHT , IMG_WIDTH, 1])
       ## cv2.imshow('1',test_image)
       ## cv2.waitKey()
##        imshow(test_mask.squeeze().astype(np.uint8))
##        plt.show()
##        test_image = np.reshape(test_image, [1,IMG_HEIGHT , IMG_WIDTH, IMG_CHANNELS])
##        test_mask = sess.run([logit],feed_dict={X: test_image})
##        test_mask = np.reshape(test_mask, [IMG_HEIGHT , IMG_WIDTH, 1])
##        for i in range(IMG_HEIGHT):
##            for j in range(IMG_WIDTH):
##                test_mask[i][j] = round(sigmoid(test_mask[i][j]))*255
##        imshow(test_mask.squeeze().astype(np.uint8))
##        plt.show()
##        time = 0
################
    batch_X, batch_Y = next_batch(batch_size, batch_count)
    batch_count += 1    
    if(batch_count >= len(X_train)/batch_size):
        batch_count = 0  

   ## if batch_count == 0:
   ##     rate = rate/2
    feed_dict = {X: batch_X, Y_: batch_Y, lr: rate,keep_prob:0.1}
    loss_value, _ = sess.run([loss, optimizer], feed_dict=feed_dict)
    train_loss_results.append(loss_value)
    tf.summary.histogram("loss", loss_value)

    if batch_count == 0:
        p = np.random.permutation(len(X_test))
        images = X_test[p]
        labels = Y_test[p]
        # only use random subset cuz of memory restrictions
        feed_dict = {X: images[0:16], Y_: labels[0:16],keep_prob:0.1}
        val_loss = sess.run([loss], feed_dict=feed_dict)
        val_epoch_results.append(val_loss)
        train_epoch_results.append(np.mean(train_loss_results))
        print(str(display_count) + " training loss:", str(np.mean(train_loss_results)), " validation loss:", str(val_loss))
        display_count += 1
        train_loss_results = []


    ##if loss_value<0.2:
     ##   break
     ##   break
##############################################################



x = np.linspace(1, num_epochs, num_epochs)
plt.plot(x, train_epoch_results, 'r')
plt.plot(x, val_epoch_results, 'b')
plt.xlabel('epochs')
plt.ylabel('loss values')
plt.show()

test_image = X_test[3]
test_mask = Y_test[3]
test_image = np.reshape(test_image, [IMG_HEIGHT , IMG_WIDTH, 1])
test_mask = np.reshape(test_mask, [IMG_HEIGHT , IMG_WIDTH, 1])
imshow(test_image.squeeze().astype(np.uint8))
plt.show()
imshow(test_mask.squeeze().astype(np.uint8))
plt.show()
test_image = np.reshape(test_image, [1,IMG_HEIGHT , IMG_WIDTH, IMG_CHANNELS])
test_mask = sess.run([logit],feed_dict={X: test_image})
test_mask = np.reshape(test_mask, [IMG_HEIGHT , IMG_WIDTH, 1])

##for i in range(IMG_HEIGHT):
##     for j in range(IMG_WIDTH):
##        if sigmoid(test_mask[i][j]) > 0.7:
##            test_mask[i][j] = 255
##        else:
##            test_mask[i][j] = 0
##imshow(test_mask.squeeze().astype(np.uint8))
##plt.show()


for i in range(IMG_HEIGHT):
     for j in range(IMG_WIDTH):
        test_mask[i][j] = round(sigmoid(test_mask[i][j]))*255
imshow(test_mask.squeeze().astype(np.uint8))
plt.show()




