# src/models.py

from tensorflow.keras import layers, models

def build_cnn_model(input_shape=(128, 128, 1)):
    """
    Builds and returns a simple CNN model for binary classification
    (cancer vs normal)
    """

    model = models.Sequential()

    # First Convolution Layer
    model.add(layers.Conv2D(32, (3,3), activation='relu', input_shape=input_shape))
    model.add(layers.MaxPooling2D((2,2)))

    # Second Convolution Layer
    model.add(layers.Conv2D(64, (3,3), activation='relu'))
    model.add(layers.MaxPooling2D((2,2)))

    # Third Convolution Layer
    model.add(layers.Conv2D(128, (3,3), activation='relu'))
    model.add(layers.MaxPooling2D((2,2)))

    # Flatten Layer
    model.add(layers.Flatten())

    # Fully Connected Layer
    model.add(layers.Dense(128, activation='relu'))

    # Output Layer (Binary Classification)
    model.add(layers.Dense(1, activation='sigmoid'))

    return model