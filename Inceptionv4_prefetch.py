import os
import keras
from cycler import K
from keras import layers,datasets,models,activations,metrics,Input
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, AveragePooling2D, Layer, BatchNormalization, \
    Activation, Dropout, Input, concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
import matplotlib.pyplot as plt
import datetime
import np_utils
from tensorflow.keras.utils import to_categorical
from numpy import asarray
from PIL import Image
from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()


gpus = tf.config.experimental.list_physical_devices('GPU')
print("Num GPUs Available:", len(gpus))

if len(gpus) > 0:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
strategy = tf.distribute.MirroredStrategy()


# Create TensorFlow datasets
def create_dataset(directory,type, batch_size):
    dataset = tf.keras.utils.image_dataset_from_directory(
        directory+type,
        labels='inferred',
        label_mode="int",
        batch_size=batch_size,
        image_size=(299,299),
    )

    return dataset

directory="/home/aisha/deeplearning/Imagenet"
train_dataset = create_dataset(directory,'/train', batch_size=256)

test_dataset = create_dataset(directory,'/test', batch_size=256)
#Preprocessing
data_aug = keras.Sequential([
    layers.Rescaling(1./255),
    layers.RandomCrop(299,299)
])
def subtract_mean(image):
    image= image.astype('float32')
    mean = tf.reduce_mean(image,axis = [1,2], keepdims=True)
    centered_image = image - mean
    return centered_image
#mapping
AUTOTUNE = tf.data.AUTOTUNE
train_aug = train_dataset.map(
    lambda x,y:
    (data_aug(x), y),
    num_parallel_calls=AUTOTUNE)
train_aug = train_aug.map(lambda x,y:
                          (subtract_mean(x),y), num_parallel_calls=AUTOTUNE)
test_aug = test_dataset.map(
    lambda x,y:
    (data_aug(x), y),
    num_parallel_calls=AUTOTUNE)
test_aug= test_aug.map(lambda x,y:
                          (subtract_mean(x),y), num_parallel_calls=AUTOTUNE)

#prefetch
train_aug = train_aug.prefetch(buffer_size=AUTOTUNE)
test_aug = test_aug.prefetch(buffer_size=AUTOTUNE)




dir_name = 'Learning_log'


def make_Tensorboard_dir(dir_name):
    root_logdir = os.path.join(os.curdir, dir_name)
    sub_dir_name = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    return os.path.join(root_logdir, sub_dir_name)


def conv_block(x, num_filter, num_row, num_col, padding='same', strides=(1, 1), use_bias=False):
    x = Conv2D(num_filter, kernel_size=(num_row, num_col), strides=strides, padding=padding, use_bias=use_bias)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x


def stem(input):
    x = conv_block(input, 32, 3, 3, strides=(2, 2), padding='valid')
    x = conv_block(x, 32, 3, 3, padding='valid')
    x = conv_block(x, 64, 3, 3)

    x1 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid')(x)
    x2 = conv_block(x, 96, 3, 3, strides=(2, 2), padding='valid')

    x = concatenate([x1, x2])

    x_1 = conv_block(x, 64, 1, 1)
    x_1 = conv_block(x_1, 96, 3, 3, padding='valid')

    x_2 = conv_block(x, 64, 1, 1)
    x_2 = conv_block(x_2, 64, 7, 1)
    x_2 = conv_block(x_2, 64, 1, 7)
    x_2 = conv_block(x_2, 96, 3, 3, padding='valid')

    x = concatenate([x_1, x_2])

    x__1 = conv_block(x, 192, 3, 3, strides=(2, 2), padding='valid')
    x__2 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid')(x)

    con = concatenate([x__1, x__2], axis=-1)

    return con


def inception_a(input):
    a1 = conv_block(input, 64, 1, 1)
    a1 = conv_block(a1, 96, 3, 3)
    a1 = conv_block(a1, 96, 3, 3)

    a2 = conv_block(input, 64, 1, 1)
    a2 = conv_block(a2, 96, 3, 3)

    a3 = conv_block(input, 96, 1, 1)

    a4 = AveragePooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(input)
    a4 = conv_block(a4, 96, 1, 1)

    con = concatenate([a1, a2, a3, a4])

    return con


def inception_b(input):
    b1 = conv_block(input, 192, 1, 1)
    b1 = conv_block(b1, 192, 1, 7)
    b1 = conv_block(b1, 224, 7, 1)
    b1 = conv_block(b1, 224, 1, 7)
    b1 = conv_block(b1, 256, 7, 1)

    b2 = conv_block(input, 192, 1, 1)
    b2 = conv_block(b2, 224, 1, 7)
    b2 = conv_block(b2, 256, 7, 1)

    b3 = conv_block(input, 384, 1, 1)

    b4 = AveragePooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(input)
    b4 = conv_block(b4, 128, 1, 1)

    con = concatenate([b1, b2, b3, b4])

    return con


def inception_c(input):
    c1 = conv_block(input, 384, 1, 1)
    c1 = conv_block(c1, 448, 1, 3)
    c1 = conv_block(c1, 512, 3, 1)
    c1_1 = conv_block(c1, 256, 1, 3)
    c1_2 = conv_block(c1, 256, 3, 1)

    c2 = conv_block(input, 384, 1, 1)
    c2_1 = conv_block(c2, 256, 1, 3)
    c2_2 = conv_block(c2, 256, 3, 1)

    c3 = conv_block(input, 256, 1, 1)

    c4 = AveragePooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(input)
    c4 = conv_block(c4, 256, 1, 1)

    con = concatenate([c1_1, c1_2, c2_1, c2_2, c3, c4])

    return con


def reduction_a(input):
    r1 = conv_block(input, 192, 1, 1)
    r1 = conv_block(r1, 224, 3, 3)
    r1 = conv_block(r1, 256, 3, 3, strides=(2, 2), padding='valid')

    r2 = conv_block(input, 384, 3, 3, strides=(2, 2), padding='valid')

    r3 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid')(input)

    con = concatenate([r1, r2, r3])

    return con


def reduction_b(input):
    r1 = conv_block(input, 256, 1, 1)
    r1 = conv_block(r1, 256, 1, 7)
    r1 = conv_block(r1, 320, 7, 1)
    r1 = conv_block(r1, 320, 3, 3, strides=(2, 2), padding='valid')

    r2 = conv_block(input, 192, 1, 1)
    r2 = conv_block(r2, 192, 3, 3, strides=(2, 2), padding='valid')

    r3 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid')(input)

    con = concatenate([r1, r2, r3])

    return con


with strategy.scope():
    def create_inception_v4():
        input_data = Input(shape=(299, 299, 3))

        x = stem(input_data)

        # 4 x inception-A
        for _ in range(4):
            x = inception_a(x)

        x = reduction_a(x)

        for _ in range(7):
            x = inception_b(x)

        x = reduction_b(x)

        for _ in range(3):
            x = inception_c(x)

        x = AveragePooling2D(pool_size=(8, 8), padding='valid')(x)
        x = Dropout(0.2)(x)
        x = Flatten()(x)

        output = Dense(999, activation='softmax')(x)

        model = Model(inputs=input_data, outputs=output, name='Inception-v4')

        return model


    def lr_schedule(epoch, lr):
        if (epoch % 2 == 0) and (epoch != 0):
            return lr * 0.96
        else:
            return lr


    def set_model(self, model):
        self.model = model


    def on_epoch_begin(self, epoch, logs=None):
        lr = K.eval(self.model.optimizer._decayed_lr(tf.float64))
        print("\n**lr at epoch{} is {}".format(epoch, lr))
    def main():
        TB_log_dir = make_Tensorboard_dir(dir_name)
        TensorB = TensorBoard(log_dir=TB_log_dir)
        model = create_inception_v4()
        model.summary()
        # Optimizer setting
        sgd = tf.keras.optimizers.SGD(learning_rate=0.045, momentum=0.9)
        rmsP = tf.keras.optimizers.RMSprop(learning_rate=0.045, epsilon=1.0)
        lr_callback = tf.keras.callbacks.LearningRateScheduler(lr_schedule)
        model.compile(optimizer=rmsP, loss='sparse_categorical_crossentropy', metrics=['accuracy',metrics.SparseTopKCategoricalAccuracy(k=5)])



        # Define checkpoint callback
        checkpoint_path = '/home/aisha/checkpoints/inceptionv4-{epoch:02d}.h5'
        checkpoint_callback = ModelCheckpoint(filepath=checkpoint_path,
                                              save_best_only=True,
                                              save_weights_only=False,
                                              mode='auto',
                                              verbose=1)


        # Train the model
        history = model.fit(train_aug,
                            epochs=50,
                            batch_size=256,
                            verbose = 1,
                            callbacks=[TensorB, checkpoint_callback])


        # Save the final model
        model.save('/home/aisha/inceptionv4.h5')
        print('Model Saved')
        plt.title("Result")
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['sparse_top_k_categorical_accuracy'])
        plt.show()

        # Score
        results = model.evaluate(test_aug)
        print('Test Loss: ', results[0])
        print('Test Accuracy: ', results[1])
        print('Test Top-5: ', results[2])
        results = model.evaluate(train_aug)
        print('Train Loss: ', results[0])
        print('Train Accuracy: ', results[1])
        print('Train Top-5: ', results[2])



    if __name__ == '__main__':
        main()
