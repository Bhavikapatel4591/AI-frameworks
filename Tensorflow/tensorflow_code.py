# tensorflow

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from time import time

mnist=tf.keras.datasets.mnist
(x_train,y_train),(x_test,y_test)= mnist.load_data()
plt.imshow(x_train[0],cmap="gray")
plt.show()
time0 = time()

x_train=tf.keras.utils.normalize(x_train,axis=1)
x_test=tf.keras.utils.normalize(x_test,axis=1)
# build model
model=tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128,activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(128,activation=tf.nn.relu))
#output layer
model.add(tf.keras.layers.Dense(10,activation=tf.nn.softmax))
#compile model
model.compile(optimizer="adam",loss= 'sparse_categorical_crossentropy',metrics=['accuracy'])
model.fit(x=x_train,y=y_train,epochs=5)
print("\nTraining Time (in minutes) =",(time()-time0)/60)
#evaluate model
model.save_weights("model_tensorflow.h5")
test_loss,test_acc=model.evaluate(x=x_test,y=y_test)
print(" Test accuracy:",test_acc)
predictions=model.predict([x_test])
print(np.argmax(predictions[1000]))
plt.imshow(x_test[1000],cmap="gray")
plt.show()