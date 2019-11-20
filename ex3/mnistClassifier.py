from keras.datasets import mnist
from keras.preprocessing.image import load_img, array_to_img
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout
from keras.optimizers import SGD, Adam
from keras import backend as K
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


CONVNET = True
# CONVNET = False


def plotHistory(hist):
    plt.subplot(211)
    plt.title("Accuracy vs. epochs")
    plt.plot(hist.history['acc'], label='acc')
    plt.plot(hist.history['val_acc'], label='test acc')
    plt.grid('on')
    plt.legend()

    plt.subplot(212)
    plt.title("Loss vs. epochs")
    plt.plot(hist.history['loss'], label='loss')
    plt.plot(hist.history['val_loss'], label='test loss')
    plt.grid('on')
    plt.legend()
    plt.show()


def linearModel():
    '''
    Simple linear model, 1 fc layers. optimizer: gradient descent on a cross-entropy loss to
    learn MNIST.
    :return: compiled model.
    '''
    model = Sequential()
    model.add(Dense(10, input_shape=(784,), activation='softmax'))

    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
    # print(model.summary())
    return model


def mlpModel(drop):
    '''
    A multi-layer perceptron (MLP), the net consisting of 4 fully-connected layers, interleaved
    with activation functions 'relu'. optimizer: SGD
    :return: compiled model.
    '''
    model = Sequential()
    model.add(Dense(392, activation='relu', input_shape=(784,)))
    model.add(Dropout(drop))
    model.add(Dense(100, activation='relu'))
    model.add(Dropout(drop))
    model.add(Dense(10, activation='softmax'))

    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
    # print(model.summary())
    return model


def convnetModel():
    '''
    Convnet, made from convolution layers, activation layers and pooling layers, ending
    with a fully connected layers before the network’s output.
    :return: compiled model.
    '''
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(5, 5), input_shape=(1, 28, 28), padding='same', activation='relu'))
    model.add(MaxPool2D((2, 2)))
    model.add(Conv2D(64, kernel_size=(5, 5), padding='same', activation='relu'))
    model.add(MaxPool2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dense(10, activation='softmax'))

    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
    # print(model.summary())
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

    return model

def autoencoders():
    model = Sequential()

    model.add(Dense(350, activation='relu', input_shape=(784,)))
    model.add(Dense(150, activation='relu'))
    model.add(Dense(30, activation='relu'))
    model.add(Dense(2, activation='relu'))
    model.add(Dense(30, activation='relu'))
    model.add(Dense(150, activation='relu'))
    model.add(Dense(350, activation='relu'))
    model.add(Dense(784, activation='sigmoid'))

    adam_opt = Adam(beta_2=0.9)
    model.compile(loss='mean_squared_error', optimizer=adam_opt, metrics=['accuracy'])

    return model


def trainingTheModel(x_train, y_train, x_test, y_test):
    '''
    Trains the model and plot the result.
    '''
    # model = convnetModel()
    # model = linearModel()
    model = mlpModel()

    history = model.fit(x_train, y_train, epochs=20, validation_data=(x_test, y_test), batch_size=50)
    score = model.evaluate(x_test, y_test, batch_size=50)
    print("score : {}".format(score))
    plotHistory(history)


def dropoutHyperParameter(drop):

    for d in drop:
        model = mlpModel(d)
        history = model.fit(x_train, y_train, epochs=20, validation_data=(x_test, y_test), batch_size=50)
        score = model.evaluate(x_test, y_test, batch_size=50)
        print("score : {}".format(score))
        plt.subplot(211)
        plt.title("Accuracy vs. epochs")
        plt.plot(history.history['acc'], label='acc d={}'.format(d))
        plt.plot(history.history['val_acc'], label='test acc d={}'.format(d))
        plt.grid('on')
        plt.legend()

        plt.subplot(212)
        plt.title("Loss vs. epochs")
        plt.plot(history.history['loss'], label='loss d={}'.format(d))
        plt.plot(history.history['val_loss'], label='test loss d={}'.format(d))
        plt.grid('on')
        plt.legend()
    plt.show()


def pcaVsAutoencoders(z, z_labels):
    model = autoencoders()
    history = model.fit(x_train, x_train, epochs=20, validation_data=(x_test, x_test), batch_size=50)
    plt.title("Loss vs. epochs")
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='test loss')
    plt.grid('on')
    plt.legend()
    plt.show()

    get_mid_output = K.function([model.layers[0].input], [model.layers[3].output])
    mid_output = get_mid_output([z])[0]
    print(mid_output)

    plt.scatter(mid_output[:, 0], mid_output[:, 1], c=z_labels)
    plt.show()

    # ********************  PCA **************************
    pca = PCA(n_components=2, whiten=False)
    data = pca.fit_transform(z)
    plt.scatter(data[:, 0], data[:, 1], c=z_labels)
    plt.show()
    print(data)



if __name__ == '__main__':

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    # plt.imshow(x_train[0], cmap='gray')
    # plt.show()

    z_labels = y_test[:5000]
    # ************** Data Pre-Processing *************************************
    if CONVNET:
        x_train = x_train.reshape(60000, 1, 28, 28).astype('float32') / 255
        x_test = x_test.reshape(10000, 1, 28, 28).astype('float32') / 255
    else:
        x_train = x_train.reshape(60000, 784).astype('float32') / 255
        x_test = x_test.reshape(10000, 784).astype('float32') / 255

    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)
    z = x_test[:5000]

    # ************** Training the Model ***************************************
    # trainingTheModel(x_train, y_train, x_test, y_test)

    # ******************** MLP check dropout hyper-parameter ********************************************
    # drop = [0.1, 0.3, 0.6, 0.9]
    # dropoutHyperParameter(drop)

    # ******************** PCA & Autoencoders*********************************************
    # pcaVsAutoencoders(z, z_labels)




