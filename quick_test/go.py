import cv2
import os
from matplotlib import pyplot as plt
import numpy as np

# 資料夾路徑
save_folder_path = "fix_img3"

mask1 = cv2.imread("output.png", 0)
mask1 = cv2.resize(mask1, (140, 50))
mask1 = 255 - mask1 

# 目標寬度和高度
target_width = 140
target_height = 50

# 讀取圖片
file_names = ["a1.jpg", "a2.jpg", "a3.jpg"]
s = input("\nchoice picture: ")
file_name = file_names[int(s)]

image = cv2.imread(file_name, 0)

resized_image = cv2.resize(image, (target_width*5, target_height*5))
mask1 = cv2.resize(mask1, (target_width*5, target_height*5))

denoised_image = resized_image

_, denoised_image = cv2.threshold(denoised_image, 0, 255, cv2.THRESH_OTSU)
denoised_image = cv2.inpaint(denoised_image, mask1, 2, cv2.INPAINT_TELEA)
kernel1 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 8))
kernel = np.ones((8, 3), dtype=np.uint8)
kernel[1, :] = 0

denoised_image = cv2.dilate(denoised_image, kernel1, iterations=3)
denoised_image = cv2.erode(denoised_image, kernel, iterations=4)

denoised_image = cv2.dilate(denoised_image, kernel1, iterations=3)
denoised_image = cv2.erode(denoised_image, kernel, iterations=2)

denoised_image = cv2.resize(denoised_image, (target_width, target_height))

# 調整圖片顏色
final_image = denoised_image

# 儲存調整後的圖片
cv2.imwrite("predict.jpg", final_image)

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dense 
from tensorflow.keras.layers import Reshape
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D,MaxPooling2D,Dense,Flatten
import random
#     return x
def create_model():
    model = Sequential()
    model.add(Conv2D(input_shape=(50, 140, 1),
                     filters=64, kernel_size=(3, 3),
                     activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(filters=128, kernel_size=(3, 3),
                     activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense( 512, activation='relu'))
    model.add(Dense( 144, activation='softmax'))  # 修改输出形状
    model.add(Reshape((4, 36))) # 添加Reshape层以修改输出形状
    return model
model = create_model()

try:
    # 加載模型
    model.load_weights('my_model.h5')
    print('load pre_train model')
except:
    print('error')
    pass
# 重新訓練模型
images = []
image = cv2.imread(file_name, 0)

image = cv2.resize(image, (target_width, target_height))
images.append(image)
x_train_image = np.array(images)

decoding_dict = {
    0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9',
    10: 'A', 11: 'B', 12: 'C', 13: 'D', 14: 'E', 15: 'F', 16: 'G', 17: 'H', 18: 'I',
    19: 'J', 20: 'K', 21: 'L', 22: 'M', 23: 'N', 24: 'O', 25: 'P', 26: 'Q', 27: 'R',
    28: 'S', 29: 'T', 30: 'U', 31: 'V', 32: 'W', 33: 'X', 34: 'Y', 35: 'Z'
}
x_train=x_train_image.astype('float32') 
x_train_normalize = x_train / 255

probability = model.predict(x_train_normalize)
prediction=tf.argmax(probability, axis=-1).numpy() 

# 將編碼轉換為原始類別標籤
prediction_decoded = []
for label in prediction:
    decoded_label = ''.join([decoding_dict[code] for code in label])
    prediction_decoded.append(decoded_label)

print("\nAns: ", prediction_decoded[0])
from PIL import Image

# 打开图片文件
image = Image.open(file_name)

# 显示图片
image.show()
