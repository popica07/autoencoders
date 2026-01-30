import numpy as np
from tensorflow.keras.datasets import cifar10
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
import math

#exercise 1

# load dataset
(x_train_full, _), (x_test_full, _) = cifar10.load_data()

# merge train+test
X = np.concatenate((x_train_full, x_test_full), axis=0)

# normalize pixel values to between -1 and 1 , as it was mentioned (advice from the class)
X = X.astype('float32') / 127.5 - 1.0

# spliting the data
X_train, X_temp = train_test_split(X, test_size=0.2, random_state=42, shuffle=True)
X_val, X_test = train_test_split(X_temp, test_size=0.5, random_state=42, shuffle=True)

# display dataset sizes to verify
# print("Train set:", X_train.shape)
# print("Validation set:", X_val.shape)
# print("Test set:", X_test.shape)


# the functuon used computing the size of latent space representation
def conv2d_output_size(W, K, P, S, C):
    W_out = math.floor((W - K + 2*P) / S) + 1.  #computation of the 2dconvolutional layer representation
    total = (W_out ** 2) * C
    return total


# building the model 
def build_cae(input_shape=(32, 32, 3)):#input initial image 32x32 3 channels
    inputs = layers.Input(shape=input_shape)#create the input 

    # encoder
    # padding same and activation relu between layers 
    x = layers.Conv2D(8, kernel_size=3, padding="same", activation="relu")(inputs)  
    x = layers.MaxPooling2D(pool_size=2)(x)                                          

    x = layers.Conv2D(12, kernel_size=3, padding="same", activation="relu")(x)       
    x = layers.MaxPooling2D(pool_size=2)(x)                                          

    latent = layers.Conv2D(16, kernel_size=3, padding="same", activation="relu")(x)  


    # decoder
    x = layers.UpSampling2D(size=2)(latent)                                          
    
    x = layers.Conv2D(12, kernel_size=3, padding="same", activation="relu")(x)      
    x = layers.UpSampling2D(size=2)(x)                                               

    outputs = layers.Conv2D(3, kernel_size=3, padding="same", activation="tanh")(x)  

    cae = models.Model(inputs, outputs, name="CAE_CIFAR10_spec")

    cae.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="mse"
    )
    return cae

cae = build_cae()

#training 
EPOCHS = 15           #minimum 10 epochs , i chose 15
BATCH  = 128


# model checkpoint ifation valid loss improves
ckpt = ModelCheckpoint(
    filepath="cae_best.keras",
    monitor="val_loss",
    save_best_only=True,
    mode="min",
    verbose=1
)

# stabilizing training before cutting of (10 epochs)
plateau = ReduceLROnPlateau(
    monitor="val_loss", factor=0.5, patience=3, min_lr=1e-5, verbose=1
)

# to stop training after no progress in 10 epochs, like verification
early = EarlyStopping(
    monitor="val_loss", patience=10, restore_best_weights=True, verbose=1
)

history = cae.fit(
    X_train, X_train,                 # autoencoder which input = target
    validation_data=(X_val, X_val),
    epochs=EPOCHS,
    batch_size=BATCH,
    shuffle=True,
    callbacks=[ckpt, plateau, early],
    verbose=1
)

# plot evolution of the error
plt.figure()
plt.plot(history.history["loss"], label="Train MSE")
plt.plot(history.history["val_loss"], label="Val MSE")
plt.xlabel("Epoch")
plt.ylabel("MSE loss")
plt.title("Evolution of Reconstruction Error (CAE)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()



# evaluate and report test error
best_model = tf.keras.models.load_model("cae_best.keras")

# evaluate
results = best_model.evaluate(X_test, X_test, verbose=0, return_dict=True)

# print loss on test dataset
print("\nTest Results")
print(f"loss: {results['loss']:.6f}")

# exercise 2

#latent space size
W = 8      # input widht
K = 3      # kernel size 
P = 1      
S = 1      
C = 16  
print (conv2d_output_size(W, K, P, S, C)) #1024 in this case


