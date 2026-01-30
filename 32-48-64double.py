import numpy as np
from tensorflow.keras.datasets import cifar10
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
import math
import os

# load dataset
(x_train_full, _), (x_test_full, _) = cifar10.load_data()

# merge train+test
X = np.concatenate((x_train_full, x_test_full), axis=0)

# normalize pixel values to between -1 and 1 , as it was mentioned (advice from the class)
X = X.astype('float32') / 127.5 - 1.0

# spliting the data
X_train, X_temp = train_test_split(X, test_size=0.2, random_state=42, shuffle=True)
X_val, X_test = train_test_split(X_temp, test_size=0.5, random_state=42, shuffle=True)


# the functuon used computing the size of latent space representation
def conv2d_output_size(W, K, P, S, C):
    W_out = math.floor((W - K + 2*P) / S) + 1
    total = (W_out ** 2) * C
    return total


# Building the model – SINGLE conv per block
def build_cae(input_shape=(32, 32, 3),):
    inputs = layers.Input(shape=input_shape)  # create the input 

    # encoder
    # padding is the 'same' and activation function is relu as mentioned in Lab 
    x = layers.Conv2D(32, kernel_size=3, padding="same", activation="relu")(inputs)
    x = layers.Conv2D(32, kernel_size=3, padding="same", activation="relu")(x)
    x = layers.MaxPooling2D(pool_size=2)(x)          

    x = layers.Conv2D(48, kernel_size=3, padding="same", activation="relu")(x)
    x = layers.Conv2D(48, kernel_size=3, padding="same", activation="relu")(x)
    x = layers.MaxPooling2D(pool_size=2)(x)         

    latent = layers.Conv2D(64, kernel_size=3, padding="same", activation="relu")(x)
    # latent: 8x8x64 for chans=(32,48,64)

    # decoder
    x = layers.UpSampling2D(size=2)(latent)          
    x = layers.Conv2D(48, kernel_size=3, padding="same", activation="relu")(x)
    x = layers.Conv2D(48, kernel_size=3, padding="same", activation="relu")(x)

    x = layers.UpSampling2D(size=2)(x)               
    x = layers.Conv2D(32, kernel_size=3, padding="same", activation="relu")(x)
    x = layers.Conv2D(32, kernel_size=3, padding="same", activation="relu")(x)


    outputs = layers.Conv2D(3, kernel_size=3, padding="same", activation="tanh")(x)

    cae = models.Model(inputs, outputs, name="CAE_CIFAR10_spec")
    cae.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="mse"
    )
    return cae


# Build and inspect
cae = build_cae()

EPOCHS = 15       
BATCH  = 128

# optional
MODEL_NAME = "cae_32_48_64_double"

ckpt = ModelCheckpoint(
    filepath=f"{MODEL_NAME}_best.keras",
    monitor="val_loss",
    save_best_only=True,
    mode="min",
    verbose=1
)
 
plateau = ReduceLROnPlateau(
    monitor="val_loss", factor=0.5, patience=3, min_lr=1e-5, verbose=1
)

# to stop training after no progress in 10 epochs
early = EarlyStopping(
    monitor="val_loss", patience=10, restore_best_weights=True, verbose=1
)

flag = True  # set to true for training, false to just load and evaluate


if flag:
    history = cae.fit(
        X_train, X_train,                 # autoencoder input=target
        validation_data=(X_val, X_val),
        epochs=EPOCHS,  
        batch_size=BATCH,
        shuffle=True,
        callbacks=[ckpt, plateau, early],
        verbose=1
    )

    # save training history for later visualization
    np.save(f"{MODEL_NAME}_history.npy", history.history, allow_pickle=True)

    # PLOT evolution of the error
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
    best_model = tf.keras.models.load_model(f"{MODEL_NAME}_best.keras")
else:
    # take the best model 
    best_model = tf.keras.models.load_model(f"{MODEL_NAME}_best.keras")

    # previous training history for plotting
    hist_path = f"{MODEL_NAME}_history.npy"
    if os.path.exists(hist_path):
        loaded_history = np.load(hist_path, allow_pickle=True).item()
        plt.figure()
        plt.plot(loaded_history["loss"], label="Train MSE")
        plt.plot(loaded_history["val_loss"], label="Val MSE")
        plt.xlabel("Epoch")
        plt.ylabel("MSE loss")
        plt.title("Evolution of Reconstruction Error (CAE)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
    else:
        print("No saved history found skip plot.")


# evaluate on test set (reconstruction)
results = best_model.evaluate(X_test, X_test, verbose=0, return_dict=True)

#print loss on test
print("\nTest Results")
print(f"loss: {results['loss']:.6f}")


# latent space size (for chans[2] = 64, spatial 8x8)
W, K, P, S, C = 8, 3, 1, 1, 64
print(conv2d_output_size(W, K, P, S, C))  # 4096
