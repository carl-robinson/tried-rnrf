# Create first network with Keras
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.datasets import mnist
from keras.callbacks import TensorBoard
from keras.models import model_from_yaml

def saveModel(model, savename):
    # serialize model to YAML
    model_yaml = model.to_yaml()
    with open(savename+".yaml", "w") as yaml_file:
        yaml_file.write(model_yaml)
        print("Yaml Model ", savename, ".yaml saved to disk")
    # serialize weights to HDF5
    model.save_weights(savename+".h5")
    print("Weights ", savename, ".h5 saved to disk")


# the data, shuffled and split between train and test sets
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.reshape(60000, 784)
X_test = X_test.reshape(10000, 784)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# create sequential model with two fully connected layers
model = Sequential()
model.add(Dense(100,  input_dim=784, name='fc1'))
model.add(Activation('sigmoid'))
model.add(Dense(10, name='fc2'))
model.add(Activation('softmax'))
model.summary()

# set learning parameters
learning_rate = 0.5
sgd = SGD(learning_rate)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

batch_size = 300
nb_epoch = 10
# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, 10)
Y_test = np_utils.to_categorical(y_test, 10)
tensorboard = TensorBoard(log_dir="_mnist", write_graph=False, write_images=True)
model.fit(X_train, Y_train, batch_size=batch_size, epochs=nb_epoch, verbose=1, callbacks=[tensorboard])

# print the loss and accuracy
scores = model.evaluate(X_test, Y_test, verbose=0)
print("%s: %.2f%%" % (model.metrics_names[0], scores[0]*100))
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

# save the model to system storage
saveModel(model, 'keras_tp1_2')