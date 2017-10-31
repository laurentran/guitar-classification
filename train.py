from __future__ import print_function

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense
from keras import backend as keras

train_path = 'data/train'
test_path = 'data/test'
epochs = 50
num_train = 736 
num_test = 318 
batch_size = 36
num_classes = 26

input_shape = (256, 256, 3)

model = Sequential()
model.add(Conv2D(32, (5,5), use_bias=True, input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64, (5,5)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(128, (5,5)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dense(256))
model.add(Activation('relu'))

model.add(Dropout(0.3))

model.add(Dense(num_classes))
model.add(Activation('softmax'))

model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.4,
    horizontal_flip=True
)

test_datagen = ImageDataGenerator(
    rescale=1./255
)

train_generator = train_datagen.flow_from_directory(
    train_path,
    batch_size=batch_size
)

test_generator = test_datagen.flow_from_directory(
    test_path,
    batch_size=batch_size
)

model.fit_generator(
    train_generator,
    steps_per_epoch=num_train/batch_size,
    epochs=epochs,
    validation_data=test_generator,
    validation_steps=num_train/batch_size
)

score = model.evaluate_generator(test_generator, batch_size)
model.save_weights('tf.h5')