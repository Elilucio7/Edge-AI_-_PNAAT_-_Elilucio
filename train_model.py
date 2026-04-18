import tensorflow as tf
from tensorflow import keras
from keras import layers

#Fazendo upload do mnist para treinamento da CNN 
(x_trn, y_trn), (x_tst, y_tst) = keras.datasets.mnist.load_data()


#Reshape para leitura correta dos dados
x_trn = x_trn.astype("float32") / 255
x_tst = x_tst.astype("float32") / 255

x_trn = x_trn.reshape(-1, 28, 28, 1)
x_tst = x_tst.reshape(-1, 28, 28, 1)

#Criação da rede neural
model = keras.Sequential([layers.Conv2D(26, kernel_size=(3, 3), activation="relu", input_shape=(28, 28, 1)),  
    layers.MaxPooling2D(pool_size=(2, 2)), 
    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.35),
    layers.Dense(10, activation='softmax')])

#Compilando e definindo parâmetros do modelo
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'])

#Treinando de fato
history = model.fit(
    x_trn, y_trn,
    epochs=7,
    batch_size=80,
    validation_split=0.1)


loss, acc = model.evaluate(x_tst, y_tst)
print(f"Test_accuracy: {acc:.2%}")
print(f"Test_loss: {loss:.2}")
