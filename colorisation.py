import numpy as np
from tensorflow.keras.datasets import cifar10
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
import math
import os

# load dataset1
(x_train_full, _), (x_test_full, _) = cifar10.load_data()

# merge train+test
X = np.concatenate((x_train_full, x_test_full), axis=0)

# normalize pixel values to between 0 and 1
X = X.astype('float32') / 127.5 - 1.0

# spliting the data
X_train, X_temp = train_test_split(X, test_size=0.2, random_state=42, shuffle=True)
X_val, X_test = train_test_split(X_temp, test_size=0.5, random_state=42, shuffle=True)

# formula for LTS
def conv2d_output_size(W, K, P, S, C):
    W_out = math.floor((W - K + 2*P) / S) + 1
    total = (W_out ** 2) * C
    return total

# mean between 3 pixels which gives us gray 
def to_gray1(x):
    return np.mean(x, axis=-1, keepdims=True) #operates on the channels number and calculates the mean of the 3 values(RGB) to get the brightness

#rgb preprocessing
# Xtr_in,  Xval_in,  Xte_in  = to_gray1(X_train), to_gray1(X_val), to_gray1(X_test)
# Xtr_out, Xval_out, Xte_out = X_train, X_val, X_test 

#input channels has to be 1(gray) and output channels has to be 3(rgb)


#yuv preprocessing in this case using Y as the luminance and U,V the chrominance but, these data should be transformed from RGB to Y,U,V
Xtr_yuv  = tf.image.rgb_to_yuv(X_train)
Xval_yuv = tf.image.rgb_to_yuv(X_val)
Xte_yuv  = tf.image.rgb_to_yuv(X_test)
Xtr_in,  Xval_in,  Xte_in  = Xtr_yuv[..., :1], Xval_yuv[..., :1], Xte_yuv[..., :1]   # Y only
Xtr_out, Xval_out, Xte_out = Xtr_yuv[..., 1:], Xval_yuv[..., 1:], Xte_yuv[..., 1:]   # (U,V)

#build the model

def build_cae(input_shape=(32, 32, 1)): #only 1 channel as input
    # encoder
    # padding is the 'same' and activation function is relu as mentioned in lab
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv2D(32, kernel_size=3, padding="same", activation="relu")(inputs)   
    x = layers.MaxPooling2D(pool_size=2)(x)                                         

    x = layers.Conv2D(48, kernel_size=3, padding="same", activation="relu")(x)       
    x = layers.MaxPooling2D(pool_size=2)(x)                                          

    latent = layers.Conv2D(64, kernel_size=3, padding="same", activation="relu")(x)  

    # decoder
    x = layers.UpSampling2D(size=2)(latent)                                          
    x = layers.Conv2D(48, kernel_size=3, padding="same", activation="relu")(x)       
    x = layers.UpSampling2D(size=2)(x)                                              

    outputs = layers.Conv2D(2, kernel_size=3, padding="same", activation=None)(x)  #2 channels as output

    cae = models.Model(inputs, outputs, name="CAE_CIFAR10_spec")
    cae.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss="mse",
    )
    return cae


cae = build_cae()


EPOCHS = 15          #  more than 10, we tried here too
BATCH  = 128


MODEL_NAME = "cae_32_48_64_color"

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

early = EarlyStopping(
    monitor="val_loss", patience=10, restore_best_weights=True, verbose=1
)

flag = True 

if flag:
    history = cae.fit(
        Xtr_in, Xtr_out,                 # input is the same as output
        validation_data=(Xval_in, Xval_out),
        epochs=EPOCHS,
        batch_size=BATCH,
        shuffle=True,
        callbacks=[ckpt, plateau, early],
        verbose=1
    )

    # saving history of MSE for later visualisation
    np.save(f"{MODEL_NAME}_history.npy", history.history, allow_pickle=True)

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

    # save best modes
    best_model = tf.keras.models.load_model(f"{MODEL_NAME}_best.keras")
else:

    # realoading the best model which was already saved
    best_model = tf.keras.models.load_model(f"{MODEL_NAME}_best.keras")

    # previous array with MSE saved values
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
        print("No saved history found, skip the plot.")

#see results
results = best_model.evaluate(Xte_in, Xte_out, verbose=0, return_dict=True)

print("\nTest Results")
print(f"loss: {results['loss']:.6f}")

W, K, P, S, C = 8, 3, 1, 1, 64
print(conv2d_output_size(W, K, P, S, C)) #4096


# # --- Step 4: Visualize ---
# def show_examples_rgb(inputs_gray, preds_rgb, targets_rgb, n=8, save_path=None):
#     n = int(min(n, inputs_gray.shape[0], preds_rgb.shape[0], targets_rgb.shape[0]))
#     fig, axes = plt.subplots(3, n, figsize=(2.4*n, 7))
#     for i in range(n):
#         inp_vis  = np.repeat(inputs_gray[i], 3, axis=-1)
#         pred_vis = np.clip(preds_rgb[i], 0.0, 1.0)
#         targ_vis = targets_rgb[i]
#         axes[0, i].imshow(inp_vis);  axes[0, i].set_title("Input (Gray)"); axes[0, i].axis("off")
#         axes[1, i].imshow(pred_vis); axes[1, i].set_title("Prediction");   axes[1, i].axis("off")
#         axes[2, i].imshow(targ_vis); axes[2, i].set_title("Target RGB");  axes[2, i].axis("off")
#     plt.tight_layout()
#     if save_path:
#         plt.savefig(save_path, dpi=150, bbox_inches="tight")
#         print(f"Saved examples to {save_path}")
#     plt.show()

# # --- Step 5: Predict and plot ---
# pred_sample = best_model.predict(Xte_in[:32], verbose=0)
# show_examples_rgb(Xte_in[:32], pred_sample, Xte_out[:32], n=8, save_path="colorize_rgb_examples.png")
