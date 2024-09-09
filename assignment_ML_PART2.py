from tensorflow import keras
import tensorflow as tf
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os
import numpy as np
import random
from collections import Counter

input_shape = (84, 84, 1)

def load_and_preprocess(img_path, input_shape):
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=input_shape[:2], color_mode='grayscale')
    img = tf.keras.preprocessing.image.img_to_array(img)
    img = img / 255.0
    return img

def extract_labels_from_filename(filename):
    class_label = filename.split('_')[1].split('.')[0]
    class_label = class_label.zfill(3)
    digits = [int(d) for d in class_label]
    return digits

def prepare_dataset(directory, input_shape):
    image_paths = []
    labels = []

    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.png'):
                image_path = os.path.join(root, file)
                image_paths.append(image_path)

                label = extract_labels_from_filename(file)
                labels.append(label)

    images = np.array([load_and_preprocess(p, input_shape) for p in image_paths])
    labels = np.array(labels)
    labels = np.array(labels).reshape(-1, 3)

    digit1_labels = keras.utils.to_categorical(labels[:,0], num_classes=10)
    digit2_labels = keras.utils.to_categorical(labels[:,1], num_classes=10)
    digit3_labels = keras.utils.to_categorical(labels[:,2], num_classes=10)

    train_images, test_images, train_digit1_labels, test_digit1_labels, train_digit2_labels, test_digit2_labels, train_digit3_labels, test_digit3_labels = train_test_split(
        images, digit1_labels, digit2_labels, digit3_labels, test_size=0.2, random_state=42
    )
    
    train_images, val_images, train_digit1_labels, val_digit1_labels, train_digit2_labels, val_digit2_labels, train_digit3_labels, val_digit3_labels = train_test_split(
        train_images, train_digit1_labels, train_digit2_labels, train_digit3_labels, test_size=0.25, random_state=42
    )

    return train_images, val_images, test_images, train_digit1_labels, val_digit1_labels, test_digit1_labels, train_digit2_labels, val_digit2_labels, test_digit2_labels, train_digit3_labels, val_digit3_labels, test_digit3_labels

dataset_directory = os.path.expanduser("~/Scripts/triple_mnist_combined")

train_images, val_images, test_images, train_digit1_labels, val_digit1_labels, test_digit1_labels, train_digit2_labels, val_digit2_labels, test_digit2_labels, train_digit3_labels, val_digit3_labels, test_digit3_labels = prepare_dataset(
    dataset_directory, input_shape)

assert os.path.exists('/Users/tpercival/Scripts/Fun/Assignment/model_checkpoints/new_model_macos.keras'), "FIle no"

def create_model(input_shape):
    inputs = keras.Input(shape=input_shape)

    # Initial Convolutional Layer with L2 Regularization
    x = keras.layers.Conv2D(64, (7, 7), strides=2, padding='same', kernel_regularizer=keras.regularizers.l2(0.0001))(inputs)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation('relu')(x)
    x = keras.layers.MaxPooling2D((3, 3), strides=2, padding='same')(x)

    # Residual Block 1
    shortcut = x
    x = keras.layers.Conv2D(64, kernel_size=3, strides=1, padding='same', kernel_regularizer=keras.regularizers.l2(0.0001))(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation('relu')(x)
    x = keras.layers.Dropout(0.2)(x)
    x = keras.layers.Conv2D(64, kernel_size=3, strides=1, padding='same', kernel_regularizer=keras.regularizers.l2(0.0001))(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.add([x, shortcut])
    x = keras.layers.Activation('relu')(x)
    
    # Residual Block 2
    shortcut = x
    x = keras.layers.Conv2D(128, kernel_size=3, strides=1, padding='same', kernel_regularizer=keras.regularizers.l2(0.0001))(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation('relu')(x)
    x = keras.layers.Dropout(0.2)(x)
    x = keras.layers.Conv2D(128, kernel_size=3, strides=1, padding='same', kernel_regularizer=keras.regularizers.l2(0.0001))(x)
    x = keras.layers.BatchNormalization()(x)
    shortcut = keras.layers.Conv2D(128, kernel_size=1, strides=1, padding='same', kernel_regularizer=keras.regularizers.l2(0.0001))(shortcut)
    shortcut = keras.layers.BatchNormalization()(shortcut)
    x = keras.layers.add([x, shortcut])
    x = keras.layers.Activation('relu')(x)
    x = keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
    
    # Residual Block 3
    shortcut = x
    x = keras.layers.Conv2D(256, kernel_size=3, strides=1, padding='same', kernel_regularizer=keras.regularizers.l2(0.0001))(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation('relu')(x)
    x = keras.layers.Dropout(0.2)(x)
    x = keras.layers.Conv2D(256, kernel_size=3, strides=1, padding='same', kernel_regularizer=keras.regularizers.l2(0.0001))(x)
    x = keras.layers.BatchNormalization()(x)
    shortcut = keras.layers.Conv2D(256, kernel_size=1, strides=1, padding='same', kernel_regularizer=keras.regularizers.l2(0.0001))(shortcut)
    shortcut = keras.layers.BatchNormalization()(shortcut)
    x = keras.layers.add([x, shortcut])
    x = keras.layers.Activation('relu')(x)
    x = keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
    
    # Residual Block 4
    shortcut = x
    x = keras.layers.Conv2D(512, kernel_size=3, strides=1, padding='same', kernel_regularizer=keras.regularizers.l2(0.0001))(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation('relu')(x)
    x = keras.layers.Dropout(0.2)(x)
    x = keras.layers.Conv2D(512, kernel_size=3, strides=1, padding='same', kernel_regularizer=keras.regularizers.l2(0.0001))(x)
    x = keras.layers.BatchNormalization()(x)
    shortcut = keras.layers.Conv2D(512, kernel_size=1, strides=1, padding='same', kernel_regularizer=keras.regularizers.l2(0.0001))(shortcut)
    shortcut = keras.layers.BatchNormalization()(shortcut)
    x = keras.layers.add([x, shortcut])
    x = keras.layers.Activation('relu')(x)

    # Global Average Pooling
    x = keras.layers.GlobalAveragePooling2D()(x)

    # Fully Connected Layer with Dropout and L2 Regularization
    x = keras.layers.Dense(512, activation='relu', kernel_regularizer=keras.regularizers.l2(0.0001))(x)
    x = keras.layers.Dropout(0.4)(x)

    # Output Layers for each digit (3 outputs)
    digit1_output = keras.layers.Dense(10, activation='softmax', name='digit1')(x)
    digit2_output = keras.layers.Dense(10, activation='softmax', name='digit2')(x)
    digit3_output = keras.layers.Dense(10, activation='softmax', name='digit3')(x)

    # Create model
    model = keras.Model(inputs=inputs, outputs=[digit1_output, digit2_output, digit3_output])

    # Compile the model with metrics for each output
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.0001), 
        loss={'digit1': 'categorical_crossentropy', 'digit2': 'categorical_crossentropy', 'digit3': 'categorical_crossentropy'},
        metrics={'digit1': ['accuracy'], 'digit2': ['accuracy'], 'digit3': ['accuracy']}
    )

    return model

model_ = create_model(input_shape)
model_.summary()

checkpoint_dir = "./model_checkpoints"

if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)

checkpoint_callback = keras.callbacks.ModelCheckpoint(
    filepath = os.path.join(checkpoint_dir, "model_epoch_{epoch:02d}.keras"),
    save_freq = 'epoch',
    save_best_only = True,
    monitor = 'val_loss',
    mode = 'min',
    verbose = 1
)

early_stopping_callback = keras.callbacks.EarlyStopping(
    monitor = 'val_loss',
    patience = 5,
    mode = 'min',
    verbose = 1,
    restore_best_weights = True,
    min_delta = 0.001
)


lr_scheduler = keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=2,
    verbose=1,
    min_lr=1e-6
)

history_ = model_.fit(
    x = train_images,
    y = {'digit1': train_digit1_labels, 'digit2': train_digit2_labels, 'digit3': train_digit3_labels},
    batch_size = 32,
    epochs = 50,
    validation_data = (val_images, {'digit1': val_digit1_labels, 'digit2': val_digit2_labels, 'digit3': val_digit3_labels}),
    callbacks=[checkpoint_callback, early_stopping_callback, lr_scheduler]
)

#model_ = keras.models.load_model('/Users/tpercival/Scripts/Fun/final_trained_model.keras')

model_.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.0001), 
        loss={'digit1': 'categorical_crossentropy', 'digit2': 'categorical_crossentropy', 'digit3': 'categorical_crossentropy'},
        metrics={'digit1': ['accuracy'], 'digit2': ['accuracy'], 'digit3': ['accuracy']}
    )

model_.save("final_trained_model.keras")

num_samples = 10  # Number of samples to evaluate
for i in range(num_samples):
    random_idx = random.randint(0, len(test_images) - 1)
    
    true_digit1 = np.argmax(test_digit1_labels[random_idx])
    true_digit2 = np.argmax(test_digit2_labels[random_idx])
    true_digit3 = np.argmax(test_digit3_labels[random_idx])
    true_number = f"{true_digit1}{true_digit2}{true_digit3}"
    
    # Get predictions
    predicted_labels = model_.predict(test_images[random_idx].reshape(1, 84, 84, 1))
    
    # Print raw predictions for debugging
    print(f"Raw predictions for sample {i}: {predicted_labels}")
    
    predicted_digit1 = np.argmax(predicted_labels[0], axis=-1)
    predicted_digit2 = np.argmax(predicted_labels[1], axis=-1)
    predicted_digit3 = np.argmax(predicted_labels[2], axis=-1)
    predicted_number = f"{predicted_digit1}{predicted_digit2}{predicted_digit3}"
    
    # Display the image with true and predicted labels
    plt.figure()
    plt.imshow(test_images[random_idx].reshape(84, 84), cmap='gray')
    plt.title(f'True Label: {true_number} | Predicted Label: {predicted_number}')
    plt.axis('off')
    plt.show()

    print(f"True number: {true_number}")
    print(f"Predicted number: {predicted_number}")

# Evaluate the model on the entire test set to check overall performance
test_loss, test_digit1_loss, test_digit2_loss, test_digit3_loss, test_digit1_accuracy, test_digit2_accuracy, test_digit3_accuracy = model_.evaluate(test_images, [test_digit1_labels, test_digit2_labels, test_digit3_labels])

print(f"Test Digit 1 Accuracy: {test_digit1_accuracy:.4f}")
print(f"Test Digit 2 Accuracy: {test_digit2_accuracy:.4f}")
print(f"Test Digit 3 Accuracy: {test_digit3_accuracy:.4f}")
