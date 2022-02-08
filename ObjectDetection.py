import tensorflow as tf
from tensorflow.keras import models, layers
import matplotlib.pyplot as plt
import numpy as np

IMG_HEIGHT = 400
IMG_WIDTH = 600
BATCH_SIZE = 20
CHANNELS = 3
EPOCHS = 50

# callback class

class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if logs.get('accuracy') > 0.95:
            print('\nReached 95% accuracy so cancelling training!')
            self.model.stop_training = True
callbacks = myCallback()

# input

dataset = tf.keras.preprocessing.image_dataset_from_directory(r'D:\CODES\Python Projects\Files\Tensorflow\wheat_leaf',
                                                    shuffle=True,
                                                    image_size=(IMG_WIDTH,IMG_HEIGHT),
                                                    batch_size= BATCH_SIZE
                                                    )

n_classes = len(dataset.class_names)
class_names = dataset.class_names

# for image_batch, label_batch in dataset.take(1):
#    print(image_batch[0].numpy())

# image visualisation

# plt.imshow(image_batch[0].numpy().astype('uint8'))
# plt.title(dataset.class_names[label_batch[0]])
# plt.axis('off')

# train test split

def get_dataset_partiions(ds, train_split=0.8, val_split=0.1, test_split=0.1, shuffle=True, shuffle_size=10000):
  
      ds_size = len(ds)
    
      if shuffle:
        ds = ds.shuffle(shuffle_size, seed=12)
    
      train_size = int(train_split * ds_size)
      val_size = int(val_split * ds_size)
    
      train_ds = ds.take(train_size)
    
      val_ds = ds.skip(train_size).take(val_size)
      test_ds = ds.skip(train_size).skip(val_size)
    
      return train_ds, val_ds, test_ds

train, validation, test = get_dataset_partiions(dataset)

# Optimizing the dataset by using cache and prefetch

train = train.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
validation = validation.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
test = test.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)

# preprocessing

rescale_and_resize = tf.keras.Sequential([
                     layers.experimental.preprocessing.Resizing(IMG_WIDTH, IMG_HEIGHT),
                     layers.experimental.preprocessing.Rescaling(1.0/255)
])

data_augmentation = tf.keras.Sequential([
                                         layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
                                         layers.experimental.preprocessing.RandomRotation(0.2)
])

model = models.Sequential([
                          rescale_and_resize,
                          data_augmentation,
                          layers.Conv2D(32, (3,3), activation='relu', input_shape=(BATCH_SIZE, IMG_HEIGHT, IMG_WIDTH, CHANNELS)),
                          layers.MaxPooling2D((2,2)),
                          layers.Conv2D(64, (3,3), activation='relu'),
                          layers.MaxPooling2D((2,2)),
                          layers.Conv2D(64, (3,3), activation='relu'),
                          layers.MaxPooling2D((2,2)),
                          
                          layers.Flatten(),
                          layers.Dense(64, activation='relu'),
                          layers.Dense(n_classes, activation='softmax')
])
 
model.build(input_shape=(BATCH_SIZE, IMG_HEIGHT, IMG_WIDTH, CHANNELS))

# model.summary()

model.compile(
    optimizer='adam',
    loss=tf.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics=['accuracy']
)

history = model.fit(
    train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    verbose=1,
    callbacks=[callbacks],
    validation_data=val,
)

scores = model.evaluate(test)

for image_batch, label_batch in dataset.take(1):
    first_image = image_batch[0].numpy().astype('uint8')
    first_label = label_batch[0].numpy()
      
    print('First image to predit:')
    plt.imshow(first_image)
    print("First image's actual label:", class_names[first_label])
      
    batch_prediction = model.predict(image_batch)
    print('predicted label:', class_names[np.argmax(batch_prediction[0])])
