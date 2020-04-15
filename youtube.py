from keras.layers import Input, Conv2D, Lambda, Dropout, MaxPooling2D, Conv2DTranspose, concatenate
from keras.models import Model
import os
import numpy as np
from tqdm import tqdm
import random
from skimage.io import imread, imshow
from skimage.transform import resize
import matplotlib.pyplot as plt
from pathlib import Path 

IMG_WIDTH = 128
IMG_HEIGHT = 128
IMG_CHANNELS = 3

np.random.seed = 42

TRAIN_PATH = "/Users/josephfrancis/Documents/ML/Background/Seg-Keras/data-science-bowl-2018/stage1_train/"
TEST_PATH = "/Users/josephfrancis/Documents/ML/Background/Seg-Keras/data-science-bowl-2018/stage1_test/"

train_ids = next(os.walk(TRAIN_PATH))[1]
test_ids = next(os.walk(TEST_PATH))[1]

x_train = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
y_train = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)

for n, id_ in tqdm(enumerate(train_ids), total=len(train_ids)):
    path = TRAIN_PATH + id_
    images_path = Path(path + "/images/" + id_)
    if images_path.is_file():
        img = imread(path + "/images/" + id_ + ".png")[:, :, :IMG_CHANNELS]
        img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode="constant", preserve_range=True)
        x_train[n] = img 
        mask = np.zeros((IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool) # Final mask
        for mask_file in next(os.walk(path + "/masks/"))[2]:
            mask_ = imread(path + "masks/" + mask_file)
            mask_ = resize(mask_, (IMG_HEIGHT, IMG_WIDTH), mode="constant", preserve_range=True)
            mask_ = np.expand_dims(mask_, axis=-1)
            mask = np.maximum(mask, mask_)
        
        y_train[n] = mask
        
        
x_test = np.zeros((len(TEST_PATH), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
size_test = []

for n, id_ in tqdm(enumerate(test_ids), total=len(test_ids)):
    path = TEST_PATH + id_ + "images/"
    valid_path = Path(path)
    if valid_path.is_file():
        img = imread(path + ".png")[:, :, :IMG_CHANNELS]
        size_test.append([img.shape[0], img.shape[1]])
        img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode="constant", preserve_range=True)
        x_test[n] = img
    

image_x = random.randint(0, len(train_ids))
imshow(x_train[image_x])
plt.show()
imshow(np.squeeze(y_train[image_x]))
plt.show()


inputs = Input((IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS))
s = Lambda(lambda x: x /2)(inputs)

# Contraction path
c1 = Conv2D(16, (3, 3), activation="relu", padding="same")(s)
c1 = Dropout(0.1)(c1)
c1 = Conv2D(16, (3,3), activation="relu", padding="same")(c1)
p1 = MaxPooling2D((2,2))(c1)

c2 = Conv2D(32, (3, 3), activation="relu", padding="same")(p1)
c2 = Dropout(0.1)(c2)
c2 = Conv2D(32, (3,3), activation="relu", padding="same")(c2)
p2 = MaxPooling2D((2,2))(c2)

c3 = Conv2D(64, (3, 3), activation="relu", padding="same")(p2)
c3 = Dropout(0.1)(c3)
c3 = Conv2D(64, (3,3), activation="relu", padding="same")(c3)
p3 = MaxPooling2D((2,2))(c3)

c4 = Conv2D(128, (3, 3), activation="relu", padding="same")(p3)
c4 = Dropout(0.1)(c4)
c4 = Conv2D(128, (3,3), activation="relu", padding="same")(c4)
p4 = MaxPooling2D((2,2))(c4)

c5 = Conv2D(256, (3, 3), activation="relu", padding="same")(p4)
c5 = Dropout(0.1)(c5)
c5 = Conv2D(256, (3,3), activation="relu", padding="same")(c5)

# Expansive path
u6 = Conv2DTranspose(128, (2, 2), strides=(2,2), padding="same")(c5)
u6 = concatenate([u6, c4])
c6 = Conv2D(128, (3, 3), activation="relu", padding="same")(u6)
c6 = Dropout(0.1)(c6)
c6 = Conv2D(128, (3,3), activation="relu", padding="same")(c6)

u7 = Conv2DTranspose(64, (2, 2), strides=(2,2), padding="same")(c6)
u7 = concatenate([u7, c3])
c7 = Conv2D(64, (3, 3), activation="relu", padding="same")(u7)
c7 = Dropout(0.1)(c7)
c7 = Conv2D(64, (3,3), activation="relu", padding="same")(c7)

u8 = Conv2DTranspose(32, (2, 2), strides=(2,2), padding="same")(c7)
u8 = concatenate([u8, c2])
c8 = Conv2D(32, (3, 3), activation="relu", padding="same")(u8)
c8 = Dropout(0.1)(c8)
c8 = Conv2D(32, (3,3), activation="relu", padding="same")(c8)

u9 = Conv2DTranspose(16, (2, 2), strides=(2,2), padding="same")(c8)
u9 = concatenate([u9, c1], axis=3) # Don't understand why axis 3
c9 = Conv2D(16, (3, 3), activation="relu", padding="same")(u9)
c9 = Dropout(0.1)(c9)
c9 = Conv2D(16, (3,3), activation="relu", padding="same")(c9)

ouputs = Conv2D(1, (1,1), activation="sigmoid")(c9)

model = Model(inputs=[inputs], outputs=[ouputs])
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])


#####################

from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard

callbacks = [
        EarlyStopping(monitor="val_loss", verbose=1, patience=2),
        ModelCheckpoint("/Users/josephfrancis/Documents/ML/Background/Seg-Keras/model_for_neuclei.h5", save_best_only=True, verbose=1), 
        TensorBoard(log_dir="/Users/josephfrancis/Documents/ML/Background/Seg-Keras/logs")
    ]

results = model.fit(x_train, y_train, validation_split=0.1, batch_size=16, callbacks=callbacks, epochs=25, verbose=1)

preds_train = model.predict(x_train[:int(x_train.shape[0] * 0.9)], verbose=1)
preds_val = model.predict(x_train[int(x_train.shape[0] * 0.9):], verbose=1)
preds_test = model.predict(x_test, verbose=1)

preds_train_t = (preds_train > 0.5).astype(np.uint8)
preds_val_t = (preds_val > 0.5).astype(np.uint8)
preds_test_t = (preds_test > 0.5).astype(np.uint8)

# On training samples

ix = random.randint(0, len(preds_train_t))
imshow(x_train[ix])
plt.show()
imshow(np.squeeze(y_train[ix]))
plt.show()
imshow(np.squeeze(preds_train_t[ix]))
plt.show()

# On validation samples

ix = random.randint(0, len(preds_val_t))
imshow(x_train[int(x_train.shape[0] * 0.9):][ix])
plt.show()
imshow(np.squeeze(y_train[int(y_train.shape[0] * 0.9):][ix]))
plt.show()
imshow(np.squeeze(preds_val_t[ix]))
plt.show()




































