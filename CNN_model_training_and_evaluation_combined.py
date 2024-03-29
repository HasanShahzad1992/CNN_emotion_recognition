import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from tensorflow.keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
import cv2

# Define directories for train, test, and validation sets
train_dir = "Facial Recognition Dataset/Train"
test_dir = "Facial Recognition Dataset/Test"
val_dir = "Facial Recognition Dataset/Validation"

# Image augmentation parameters
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,  # Rescale to normalize pixel values
    shear_range=0.2,
    zoom_range=0.2,
    rotation_range=30,
    width_shift_range=0.3,
    height_shift_range=0.3,
    horizontal_flip=True
)

test_datagen = ImageDataGenerator(rescale=1.0 / 255)
val_datagen = ImageDataGenerator(rescale=1.0 / 255)

# Load and augment the training dataset
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(48, 48),
    batch_size=32,
    color_mode="grayscale",
    class_mode="categorical"
)

# Load and augment the test dataset
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(48, 48),
    batch_size=32,
    color_mode="grayscale",
    class_mode="categorical"
)

# Load and augment the validation dataset
val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(48, 48),
    batch_size=32,
    color_mode="grayscale",
    class_mode="categorical"
)

# Adjust the model architecture
model = Sequential([
    Conv2D(32, (3, 3), activation='sigmoid', input_shape=(48, 48, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='sigmoid'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='sigmoid'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(256, activation='sigmoid'),
    Dropout(0.25),
    Dense(6, activation='softmax')
])

# Compile the model
model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Save the best model during training
model_checkpoint = ModelCheckpoint('best_model.keras', monitor='val_accuracy', save_best_only=True)

# Train the model with increased epochs
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    epochs=500,  # Increased number of epochs
    validation_data=val_generator,  # Use augmented validation data
    validation_steps=val_generator.samples // val_generator.batch_size,
    callbacks=[model_checkpoint]
)

# Save the original model
model.save("trained_model.keras")

# Evaluate the model
test_loss, test_accuracy = model.evaluate(test_generator)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

# Predict class probabilities for test data
y_pred_prob = model.predict(test_generator)

# Extract predicted classes
y_pred = np.argmax(y_pred_prob, axis=1)

# True classes
y_true = test_generator.classes

# Calculate evaluation metrics
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average='weighted')
recall = recall_score(y_true, y_pred, average='weighted')
f1 = f1_score(y_true, y_pred, average='weighted')
cm = confusion_matrix(y_true, y_pred)

# Display evaluation metrics and confusion matrix
print("Confusion Matrix:")
print(cm)

def test_model(image_path, test_generator):
    model = load_model("trained_model.keras")
    # Load the image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, (48, 48))
    image = np.expand_dims(image, axis=-1)
    image = image.astype('float32') / 255.0
    image = np.reshape(image, (1, 48, 48, 1))

    # Make predictions
    predictions = model.predict(image)
    predicted_class_index = np.argmax(predictions)

    # Get the class indices from the test generator
    class_indices = test_generator.class_indices

    # Get the predicted class name
    predicted_class_name = list(class_indices.keys())[predicted_class_index]

    # Print the predicted class name
    print("Predicted class:", predicted_class_name)

    # Display the image with the predicted class name using matplotlib
    plt.imshow(cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB))
    plt.title(f"Predicted class: {predicted_class_name}")
    plt.axis('off')
    plt.show()


test_model(r"Facial Recognition Dataset\Test\Angry\Angry-7.jpg", test_generator)
test_model(r"Facial Recognition Dataset\Test\Fear\Fear-7.jpg", test_generator)
test_model(r"Facial Recognition Dataset\Test\Happy\Happy-4.jpg", test_generator)
test_model(r"Facial Recognition Dataset\Test\Sad\Sad-3.jpg", test_generator)
test_model(r"Facial Recognition Dataset\Test\Neutral\Neutral-3.jpg", test_generator)
test_model(r"Facial Recognition Dataset\Test\Surprise\Suprise-3.jpg", test_generator)
