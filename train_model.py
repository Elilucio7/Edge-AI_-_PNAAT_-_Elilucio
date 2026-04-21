from tensorflow import keras
from keras import layers
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split

#Fazendo upload do mnist para treinamento da CNN 
(x_trn, y_trn), (x_tst, y_tst) = keras.datasets.mnist.load_data()

x_trn, x_val, y_trn, y_val = train_test_split(x_trn, y_trn, stratify=y_trn, test_size=0.25)


#Reshape para leitura correta dos dados
x_trn = x_trn.astype("float32") / 255
x_tst = x_tst.astype("float32") / 255

x_trn = x_trn.reshape(-1, 28, 28, 1)
x_tst = x_tst.reshape(-1, 28, 28, 1)

#Criação da rede neural
model = keras.Sequential([layers.Conv2D(30, kernel_size=(3, 3), activation="relu", input_shape=(28, 28, 1)),  
    layers.MaxPooling2D(pool_size=(3, 3)), 
    layers.Conv2D(30, kernel_size=(3, 3), activation="relu"),
    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.35),
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
    validation_data=(x_val, y_val),
    epochs=6,
    batch_size=128,
    validation_split=0.1)

keras_path = "model.keras"
model.save(keras_path) 