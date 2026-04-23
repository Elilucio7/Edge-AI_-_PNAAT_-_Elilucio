from tensorflow import keras
from keras import layers
from keras.optimizers import Adam

#Fazendo upload do mnist para treinamento da CNN 
(x_trn, y_trn), (x_tst, y_tst) = keras.datasets.mnist.load_data()

#Reshape para leitura correta dos dados
x_trn = x_trn.astype("float32") / 255
x_tst = x_tst.astype("float32") / 255

x_trn = x_trn.reshape(-1, 28, 28, 1)
x_tst = x_tst.reshape(-1, 28, 28, 1)

#Criação da rede neural
model = keras.Sequential([layers.Conv2D(20, kernel_size=(3, 3), activation="relu", input_shape=(28, 28, 1)),  
    layers.MaxPooling2D(pool_size=(3, 3)), 
    layers.Conv2D(20, kernel_size=(3, 3), activation="relu"),
    layers.Flatten(),
    layers.Dense(64, activation="relu"),
    layers.Dropout(0.15),
    layers.Dense(10, activation='softmax')])

#Compilando e definindo parâmetros do modelo
adamOptimizer = Adam(learning_rate=0.001)
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'])

#Treinando de fato
history = model.fit(
    x_trn, y_trn,
    epochs=5,
    batch_size=128,
    validation_split=0.1)

#Print da acurácia e perda do modelo com um último teste  98.769999%
loss, accuracy = model.evaluate(x_tst, y_tst)
print(f"Accuracy: {accuracy:2%}")

#Salvando o modelo em formato .h5
h5_path = "model.h5"
model.save(h5_path)