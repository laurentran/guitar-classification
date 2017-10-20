from __future__ import print_function

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense
from keras import backend as keras

im_width = 256
im_height = 256

train_path = 'reverbData/train'
test_path = 'reverbData/test'
image_gen_path = 'reverbData/generated'
epochs = 100
num_train = 647
num_test = 358
batch_size = 32

input_shape = (im_width, im_height, 3)

model = Sequential()
model.add(Conv2D(32, (5,5), use_bias=True, activation='relu', input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64, (5,5), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(128, (5,5), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.3))

model.add(Dense(16, activation='softmax'))

model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

train_datagen = ImageDataGenerator(
    rotation_range=45,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.2,
    zoom_range=0.4,
    horizontal_flip=True
)

test_datagen = ImageDataGenerator(
    rotation_range=45,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.2,
    zoom_range=0.4,
    horizontal_flip=True
)

train_generator = train_datagen.flow_from_directory(
    train_path,
    batch_size=batch_size,
    save_to_dir=image_gen_path
)

test_generator = test_datagen.flow_from_directory(
    test_path,
    batch_size=batch_size,
    save_to_dir=image_gen_path
)

model.fit_generator(
    train_generator,
    steps_per_epoch=num_train/batch_size,
    epochs=epochs,
    validation_data=test_generator,
    validation_steps=num_train/batch_size
)

score = model.evaluate_generator(test_generator, batch_size)