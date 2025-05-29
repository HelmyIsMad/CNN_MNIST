from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.datasets import mnist
from matplotlib import pyplot as plt
import numpy as np
import tensorflow as tf

def init_model():
    global model
    model = Sequential(
        [
            # Input: resized to 28x28, grayscale (1 channel)
            Conv2D(32, kernel_size=(3, 3), activation="relu", input_shape=(28, 28, 1)),
            MaxPooling2D(pool_size=(2, 2)),
            Conv2D(64, kernel_size=(3, 3), activation="relu"),
            MaxPooling2D(pool_size=(2, 2)),
            Flatten(),
            Dense(64, activation="relu"),
            Dense(10, activation="softmax"),
        ]
    )


def load_model():
    global model
    model = tf.keras.models.load_model("model.h5")


def load_data():
    global x_test, y_test, x_train, y_train
    # Load the dataset
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    print("Training set:", x_train.shape, y_train.shape)
    print("Test set:", x_test.shape, y_test.shape)

    # Normalization
    x_train = x_train / 255.0
    x_test = x_test / 255.0

    # fitting the dataset for the CNN
    x_train = x_train.reshape(-1, 28, 28, 1)
    x_test = x_test.reshape(-1, 28, 28, 1)


def train():
    train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(32)

    model.compile(
        optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )
    model.fit(train_ds, epochs=10)


def show_imgs():
    # Show the first 9 images
    plt.figure(figsize=(5, 5))
    for i in range(9):
        plt.subplot(3, 3, i + 1)
        plt.imshow(x_train[i], cmap="gray")
        plt.title(f"Label: {y_train[i]}")
        plt.axis("off")
    plt.tight_layout()
    plt.show()


def test(index):
    predictions = model.predict(x_test)
    predicted_label = np.argmax(predictions[index])
    true_label = y_test[index]
    plt.imshow(x_test[index].reshape(28, 28), cmap="gray")
    plt.title(f"Predicted: {predicted_label}, True: {true_label}")
    plt.axis("off")
    plt.show()


def first_run():
    init_model()
    load_data()
    train()
    model.save("model.h5")
    test(0)


def run():
    global x_test
    load_model()
    load_data()

    # choose an index to test with
    index = np.random.randint(0, len(x_test))
    index = 0

    # Just uncomment the function you wanna use
    # test(index)
    # testAll()


def main():
    import os
    if not os.path.exists("model.h5"):
        first_run()
    else:
        run()

def testAll():
    predictions = model.predict(x_test)
    for i in range(len(predictions)):
        predicted_label = np.argmax(predictions[i])
        true_label = y_test[i]
        plt.imshow(x_test[i].reshape(28, 28), cmap="gray")
        plt.title(f"Index: {i}, Predicted: {predicted_label}, True: {true_label}")
        plt.axis("off")
        plt.pause(1)

def testScreen():
    global model
    import pyautogui
    import keyboard
    while not keyboard.is_pressed("k"):
        screenshot = pyautogui.screenshot(region=(1450, 300, 1000, 1000))  
        img_gray = screenshot.convert('L')
        img_resized = img_gray.resize((28, 28))
        img_array = np.array(img_resized) / 255.0  
        img_input = img_array.reshape(1, 28, 28, 1)
        predictions = model.predict(img_input)
        predicted_label = np.argmax(predictions[0])

        plt.imshow(img_input.reshape(28, 28), cmap="gray")
        plt.title(f"Predicted: {predicted_label}")
        plt.axis("off")
        plt.pause(0.5)

if __name__ == "__main__":
    main()