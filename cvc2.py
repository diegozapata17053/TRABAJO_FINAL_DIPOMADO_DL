# -*- coding: utf-8 -*-

"""
Importar Librerías
"""

import numpy as np
import os
import re
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

import keras
from keras.utils import to_categorical
from keras.models import Sequential,Input,Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
#from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU

"""
Cargar set de Imágenes
"""

dirname = os.path.join(os.getcwd(), 'tipos_cancer')
imgpath = dirname + os.sep 

images = []
directories = []
dircount = []
prevRoot=''
cant=0

print("leyendo imagenes de ",imgpath)

for root, dirnames, filenames in os.walk(imgpath):
    for filename in filenames:
        if re.search("\.(jpg|jpeg|png|bmp|tiff)$", filename):
            cant=cant+1
            filepath = os.path.join(root, filename)
            image = plt.imread(filepath)
            images.append(image)
            b = "Leyendo..." + str(cant)
            print (b, end="\r")
            if prevRoot !=root:
                print(root, cant)
                prevRoot=root
                directories.append(root)
                dircount.append(cant)
                cant=0
dircount.append(cant)

dircount = dircount[1:]
dircount[0]=dircount[0]+1
print('Directorios leidos:',len(directories))
print("Imagenes en cada directorio", dircount)
print('suma Total de imagenes en subdirs:',sum(dircount))

"""
Creamos las etiquetas
"""

labels=[]
indice=0
for cantidad in dircount:
    for i in range(cantidad):
        labels.append(indice)
    indice=indice+1
print("Cantidad etiquetas creadas: ",len(labels))

tiposcancer=[]
indice=0
for directorio in directories:
    name = directorio.split(os.sep)
    print(indice , name[len(name)-1])
    tiposcancer.append(name[len(name)-1])
    indice=indice+1
    
y = np.array(labels)
X = np.array(images, dtype=np.uint8) #convierto de lista a numpy

# Encuentra los números únicos de la etiquetas a entrenar
classes = np.unique(y)
nClasses = len(classes)
print('Total numero de salidas : ', nClasses)
print('Clases de salida : ', classes)

"""
Creamos Sets de Entrenamiento y Test
"""

train_X,test_X,train_Y,test_Y = train_test_split(X,y,test_size=0.2)
print('Training data shape : ', train_X.shape, train_Y.shape)
print('Testing data shape : ', test_X.shape, test_Y.shape)

plt.figure(figsize=[5,5])

# Mostrar la primera imagen en datos de entrenamiento.
plt.subplot(121)
plt.imshow(train_X[0,:,:], cmap='gray')
plt.title("Ground Truth: {}".format(train_Y[0])) ## 

# Mostrar la primera imagen en los datos de entrenamiento.
plt.subplot(122)
plt.imshow(test_X[0,:,:], cmap='gray')
plt.title("Ground Truth : {}".format(test_Y[0])) ##Ground Truth 

"""
Preprocesamos las imagenes
"""

train_X = train_X.astype('float32')
test_X = test_X.astype('float32')
train_X = train_X / 255.
test_X = test_X / 255.

"""
Hacemos la Codificación One-Hot para la red
"""

# Cambiar las etiquetas de categóricas a una codificación en caliente (one-hot encoding)
train_Y_one_hot = to_categorical(train_Y)
test_Y_one_hot = to_categorical(test_Y)

# Mostrar el cambio para la etiqueta de categoría usando la codificación de un solo uso (one-hot encoding)
print('Etiqueta original:', train_Y[0])
print('Después de la conversión a one-hot:', train_Y_one_hot[0])

"""
Creamos el Set de Entrenamiento y Validación
"""

#Mezclar todo y crear los grupos de entrenamiento y testing
train_X,valid_X,train_label,valid_label = train_test_split(train_X, train_Y_one_hot, test_size=0.2, random_state=13)
[ ]

print(train_X.shape,valid_X.shape,train_label.shape,valid_label.shape)

########################################################################################
########################################################################################
"""
Creamos el modelo de CNN
"""

#declaramos variables con los parámetros de configuración de la red
INIT_LR = 1e-3 # Valor inicial de learning rate. El valor 1e-3 corresponde con 0.001
epochs = 6 # Cantidad de iteraciones completas al conjunto de imagenes de entrenamiento 6
batch_size = 64 # cantidad de imágenes que se toman a la vez en memoria 64

cancer_model = Sequential()
cancer_model.add(Conv2D(32, kernel_size=(3, 3),activation='linear',padding='same',input_shape=(21,28,3)))
cancer_model.add(LeakyReLU(alpha=0.1))
cancer_model.add(MaxPooling2D((2, 2),padding='same'))
cancer_model.add(Dropout(0.5))

cancer_model.add(Flatten())
cancer_model.add(Dense(32, activation='linear'))
cancer_model.add(LeakyReLU(alpha=0.1))
cancer_model.add(Dropout(0.5))
cancer_model.add(Dense(nClasses, activation='softmax'))

cancer_model.summary()

cancer_model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adagrad(lr=INIT_LR, decay=INIT_LR / 100),metrics=['accuracy'])

"""
Entrenamos el modelo: Aprende a clasificar imágenes
"""

# este paso puede tomar varios minutos, dependiendo de tu ordenador, cpu y memoria ram libre
# tarda 4 minutos
cancer_train = cancer_model.fit(train_X, train_label, batch_size=batch_size,epochs=epochs,verbose=1,validation_data=(valid_X, valid_label))

# guardamos la red, para reutilizarla en el futuro, sin tener que volver a entrenar
cancer_model.save("cancer_mnist.h5py")

"""
Evaluamos la red
"""

test_eval = cancer_model.evaluate(test_X, test_Y_one_hot, verbose=1)

print('Test loss:', test_eval[0])
print('Test accuracy:', test_eval[1])    ##accuracy(exactitud)

accuracy = cancer_train.history['acc']
val_accuracy = cancer_train.history['val_acc']
loss = cancer_train.history['loss']
val_loss = cancer_train.history['val_loss']
epochs = range(len(accuracy))
plt.plot(epochs, accuracy, 'bo', label='Training accuracy')
plt.plot(epochs, val_accuracy, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()

predicted_classes2 = cancer_model.predict(test_X)

predicted_classes=[]
for predicted_cancer in predicted_classes2:
    predicted_classes.append(predicted_cancer.tolist().index(max(predicted_cancer)))
predicted_classes=np.array(predicted_classes)

predicted_classes.shape, test_Y.shape

"""
Aprendamos de los errores: Qué mejorar
"""

correct = np.where(predicted_classes==test_Y)[0]
print("Found %d correct labels" % len(correct))
for i, correct in enumerate(correct[0:9]):
    plt.subplot(3,3,i+1)
    plt.imshow(test_X[correct].reshape(21,28,3), cmap='gray', interpolation='none')
    plt.title("{}, {}".format(tiposcancer[predicted_classes[correct]],
                                                    tiposcancer[test_Y[correct]]))

    plt.tight_layout()


incorrect = np.where(predicted_classes!=test_Y)[0]
print("Found %d incorrect labels" % len(incorrect))
for i, incorrect in enumerate(incorrect[0:9]):
    plt.subplot(3,3,i+1)
    plt.imshow(test_X[incorrect].reshape(21,28,3), cmap='gray', interpolation='none')
    plt.title("{}, {}".format(tiposcancer[predicted_classes[incorrect]],
                                                    tiposcancer[test_Y[incorrect]]))
    plt.tight_layout()


target_names = ["Class {}".format(i) for i in range(nClasses)]
print(classification_report(test_Y, predicted_classes, target_names=target_names))
