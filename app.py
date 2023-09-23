import numpy as np
import pandas as pd
from keras.layers import Dense, Conv2D, Flatten, Dropout, BatchNormalization
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def load_data():
    # Load MNIST & NIST dataset
    digits_data_train = pd.read_csv("train.csv")
    digits_data_test = pd.read_csv("test.csv")
    letters_dataset = pd.read_csv("A_Z Handwritten Data.csv")
    digits_data = pd.concat([digits_data_train, digits_data_test], ignore_index=True)
    # Rename correct label column name to label
    digits_data.rename(columns={'0':'label'}, inplace=True)
    letters_dataset.rename(columns={'0':'label'}, inplace=True)

    # Select 1000 samples from each class
    digits_data = digits_data.groupby('label').head(1000)
    letters_dataset = letters_dataset.groupby('label').head(1000)

    # Split data into X and Y for each type
    Y1 = digits_data['label']
    X1 = digits_data.drop('label', axis=1)
    Y2 = letters_dataset["label"]
    X2 = letters_dataset.drop("label", axis=1)

    # Split data into train and test set
    x_train1, x_test1, y_train1, y_test1 = train_test_split(X1, Y1, train_size=0.9)
    x_train2, x_test2, y_train2, y_test2 = train_test_split(X2, Y2, train_size=0.9)

    # Convert into numpy array to ease preprocessing
    x_train1 = x_train1.to_numpy()
    x_test1 = x_test1.to_numpy()
    x_train2 = x_train2.to_numpy()
    x_test2 = x_test2.to_numpy()

    # Convert Y into one-hot vectors
    y_train1 = to_categorical(y_train1, num_classes=36)
    y_test1 = to_categorical(y_test1, num_classes=36)
    y_train2 = to_categorical(y_train2, num_classes=36)
    y_test2 = to_categorical(y_test2, num_classes=36)

    # Reshape Xs into CNN input dimension
    x_train1 = x_train1.reshape(x_train1.shape[0], 28, 28, 1)
    x_test1 = x_test1.reshape(x_test1.shape[0], 28, 28, 1)
    x_train2 = x_train2.reshape(x_train2.shape[0], 28, 28, 1)
    x_test2 = x_test2.reshape(x_test2.shape[0], 28, 28, 1)

    # Combine each X and Y from each dataset
    x_train = np.concatenate((x_train1, x_train2), axis=0)
    x_test = np.concatenate((x_test1, x_test2), axis=0)
    y_train = np.concatenate((y_train1, y_train2), axis=0)
    y_test = np.concatenate((y_test1, y_test2), axis=0)

    return x_train, x_test, y_train, y_test

def preprocessing(x_train, x_test):
    x_train = x_train / 255
    x_test = x_test / 255
    return x_train, x_test

x_train, x_test, y_train, y_test = load_data()
(x_train, x_test) = preprocessing(x_train, x_test)

# Initialize Keras image generator
generator = ImageDataGenerator(rotation_range=10, zoom_range=0.1, width_shift_range=0.1, height_shift_range=0.1,horizontal_flip=True)

# CNN Model
model = Sequential([
    Conv2D(filters=32, kernel_size=3, input_shape=(28, 28, 1), activation="relu"),
    BatchNormalization(),
    Conv2D(filters=32, kernel_size=3, activation="relu"),
    BatchNormalization(),
    Conv2D(filters=32, kernel_size=5, strides=2, padding="same", activation="relu"),
    BatchNormalization(),
    Dropout(0.4),
    Conv2D(filters=64, kernel_size=3, activation="relu"),
    BatchNormalization(),
    Conv2D(filters=64, kernel_size=3, activation="relu"),
    BatchNormalization(),
    Conv2D(filters=64, kernel_size=5, strides=2, padding="same", activation="relu"),
    BatchNormalization(),
    Dropout(0.4),
    Conv2D(128, kernel_size=4, activation="relu"),
    BatchNormalization(),
    Flatten(),
    Dropout(0.4),
    Dense(36, activation="softmax")
])

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

history = model.fit(generator.flow(x_train, y_train, batch_size=64), epochs=50, steps_per_epoch=x_train.shape[0]//64, validation_data=(x_test, y_test))
_, accuracy = model.evaluate(x_test, y_test)
accuracy *= 100
print(f"Accuracy: {accuracy:.2f}%")

# Compute confusion matrix for all 36 labels
y_pred = model.predict(x_test)
y_pred_classes_all = np.argmax(y_pred, axis=1)
y_true_digits = np.argmax(y_test[:x_test.shape[0]], axis=1)   # Extract true labels for digits (0-9)
y_true_letters = np.argmax(y_test[x_test.shape[0]:], axis=1)   # Extract true labels for letters (10-35)

# Create a single array with all true labels (0-35)
y_true_all = np.concatenate((y_true_digits, y_true_letters + 10), axis=0)
cm = confusion_matrix(y_true_all, y_pred_classes_all)

# Visualize confusion matrix
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=list(range(36)), yticklabels=list(range(36)))
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()


# Extract training and validation accuracy and loss from the history object
train_accuracy = history.history['accuracy']
train_loss = history.history['loss']
val_accuracy = history.history['val_accuracy']  # Validation accuracy
val_loss = history.history['val_loss']  # Validation loss

# Print the final validation accuracy and loss
final_val_accuracy = val_accuracy[-1]
final_val_loss = val_loss[-1]
print(f"Final Validation Accuracy: {final_val_accuracy:.4f}")
print(f"Final Validation Loss: {final_val_loss:.4f}")

# Plot training and validation accuracy
plt.plot(range(1, len(train_accuracy) + 1), train_accuracy, label='Training Accuracy')
plt.plot(range(1, len(val_accuracy) + 1), val_accuracy, label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.show()

# Plot training and validation loss
plt.plot(range(1, len(train_loss) + 1), train_loss, label='Training Loss')
plt.plot(range(1, len(val_loss) + 1), val_loss, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()

# Save the model
model.save('model.h5')
print("Trained model saved as 'model.h5'.")