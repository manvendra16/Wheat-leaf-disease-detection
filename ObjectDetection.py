import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
import os

class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if logs.get('accuracy') > 0.95:
            print('\nReached 95% accuracy so cancelling training!')
            self.model.stop_training = True
callbacks = myCallback()

def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if img is not None:
            images.append(cv2.resize(img,(600,400),3))
    return images

train = ImageDataGenerator(rescale=1/255)
validation = ImageDataGenerator(rescale=1/255)

training = train.flow_from_directory(r'C:\Users\91979\Downloads\Dataset\Training',
                                    target_size= (400,600),
                                    batch_size= 10,
                                    class_mode= 'categorical')
validating = validation.flow_from_directory(r'C:\Users\91979\Downloads\Dataset\Validation',
                                    target_size= (400,600),
                                    batch_size= 10,
                                    class_mode= 'categorical')

model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(64, (3,3), activation='relu', input_shape=(400,600,3)),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(units=64, activation=tf.nn.relu),
        tf.keras.layers.Dense(units=3, activation=tf.nn.softmax)
])

model.summary()

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(training, epochs=10, callbacks=[callbacks], validation_data=validating)

# print(validating.class_indices)

dir_path = r'C:\Users\91979\Downloads\Dataset\Testing'
im = load_images_from_folder(dir_path)

arr = [i.reshape(1,i.shape[0],i.shape[1],i.shape[2]) for i in im]

out = [model.predict(i) for i in arr]

for i in out:
    if i[0][0] == 1:
        print('Healthy')
    elif i[0][1] == 1:
        print('Septoria')
    else:
        print('Stripe Rust')
        