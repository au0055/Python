import array
import numpy as np
from matplotlib.cbook import flatten
from numpy import float32
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from keras.datasets import cifar10
(train_images, train_labels), (test_images, test_labels)= cifar10.load_data()

from keras.utils import to_categorical
import matplotlib as plt

from tensorflow import Dense 


fashion_mnist = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
class_names={ 0: 'T-shirt/top',
              1: 'Trouser',
              2: 'Pullover',
              3: 'Dress',
              4: 'Coat',
              5: 'Sandal',
              6: 'Shirt',
              7: 'Sneaker',
              8: 'Bag',
              9: 'Ankle boot' }
plt.figure(figsize=(10,10))
for i in range(25):
   plt.subplot(5,5,i+1)
   plt.xticks([])
   plt.yticks([])
   plt.imshow(train_images[i], cmap=plt.cm.binary)
   plt.xlabel(class_names[train_labels[i]])
plt.show()

train_x, val_x, train_y, val_y = train_test_split(train_images, train_labels, stratify=train_labels, random_state=48, test_size=0.05)
(test_x, test_y)=(test_images, test_labels)

# normalize to range 0-1
train_x = train_x / 255.0
val_x = val_x / 255.0
test_x = test_x / 255.0


train_y[:5]
-> array([[2],                  (Pullover)
          [8],                  (Bag)
          [6],                  (Shirt)
          [1],                  (Trouser)
          [.3]], dtype=uint8).  (Dress) 


train_y = to_categorical(train_y)
val_y = to_categorical(val_y)
test_y = to_categorical(test_y)

array([[0., 0., 1., 0., 0., 0., 0., 0., 0., 0.],        
       [0., 0., 0., 0., 0., 0., 0., 0., 1., 0.],        
       [0., 0., 0., 0., 0., 0., 1., 0., 0., 0.],        
       [0., 1., 0., 0., 0., 0., 0., 0., 0., 0.],        
       [0., 0., 0., 1., 0., 0., 0., 0., 0., 0.]], dtype=float32)

print(train_x.shape)  #(57000, 28, 28)
print(train_y.shape)  #(57000, 10)
print(val_x.shape)    #(3000, 28, 28)
print(val_y.shape)    #(3000, 10)
print(test_x.shape)   #(10000, 28, 28)
print(test_y.shape)   #(10000, 10)

model_mlp = SequentialFeatureSelector()
model_mlp.add(flatten(input_shape=(28, 28)))
model_mlp.add(Dense(350, activation='relu'))
model_mlp.add(Dense(10, activation='softmax'))
print(model_mlp.summary())
model_mlp.compile(optimizer="adam",loss='categorical_crossentropy', metrics=['accuracy'])

early_stop=tf.keras.callbacks.EarlyStopping(monitor='val_loss', restore_best_weights= True, patience=5, verbose=1)
callback = [early_stop]

history_mlp = model_mlp.fit(train_x, train_y, epochs=100, batch_size=32, validation_data=(val_x, val_y), callbacks=callback)

# define the function:
def plot_history(hs, epochs, metric):
    plt.rcParams['font.size'] = 16
    plt.figure(figsize=(10, 8))
    for label in hs:
        plt.plot(hs[label].history[metric], label='{0:s} train {1:s}'.format(label, metric), linewidth=2)
        plt.plot(hs[label].history['val_{0:s}'.format(metric)], label='{0:s} validation {1:s}'.format(label, metric), linewidth=2)
    plt.ylim((0, 1))
    plt.xlabel('Epochs')
    plt.ylabel('Loss' if metric=='loss' else 'Accuracy')
    plt.legend()
    plt.grid()
    plt.show()
plot_history(hs={'MLP': history_mlp}, epochs=15, metric='loss')
plot_history( hs={'MLP': history_mlp}, epochs=15, metric='accuracy')


mlp_train_loss, mlp_train_acc = model_mlp.evaluate(train_x,  train_y, verbose=0)
print('\nTrain accuracy:', np.round(mlp_train_acc,3))
mlp_val_loss, mlp_val_acc = model_mlp.evaluate(val_x,  val_y, verbose=0)
print('\nValidation accuracy:', np.round(mlp_val_acc,3))
mlp_test_loss, mlp_test_acc = model_mlp.evaluate(test_x,  test_y, verbose=0)
print('\nTest accuracy:', np.round(mlp_test_acc,3))

#Output:
#Train accuracy: 0.916
#Validation accuracy: 0.889
#Test accuracy: 0.866


model_cnn = keras.Sequential()
model_cnn.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model_cnn.add(tf.keras.layers.MaxPooling2D((2, 2)))
model_cnn.add(tf.keras.layers.Flatten())
model_cnn.add(Dense(100, activation='relu'))
model_cnn.add(Dense(10, activation='softmax'))
model_cnn.compile(optimizer="adam", loss='categorical_crossentropy', metrics=['accuracy'])
print(model_cnn.summary())
history_cnn= model_cnn.fit(train_x, train_y, epochs=100, batch_size=32, validation_data=(val_x, val_y), callbacks=callback)

plot_history(hs={'CNN': history_cnn},epochs=10,metric='loss')
plot_history(hs={'CNN': history_cnn},epochs=10,metric='accuracy')


cnn_train_loss, cnn_train_acc = model_cnn.evaluate(train_x,  train_y, verbose=2)
print('\nTrain accuracy:', cnn_train_acc)
cnn_val_loss, cnn_val_acc = model_cnn.evaluate(val_x,  val_y, verbose=2)
print('\nValidation accuracy:', cnn_val_acc)
cnn_test_loss, cnn_test_acc = model_cnn.evaluate(test_x,  test_y, verbose=2)
print('\nTest accuracy:', cnn_test_acc)
#Output:
#Train accuracy: 0.938
#Validation accuracy: 0.91
#Test accuracy: 0.908
