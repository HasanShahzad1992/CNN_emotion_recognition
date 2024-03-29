import random
import numpy as np
import cv2
from deap import base, creator, tools, algorithms
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import models, layers


# Define directories for train, test, and validation sets
train_dir = "Facial Recognition Dataset/Train"
test_dir = "Facial Recognition Dataset/Test"
val_dir = "Facial Recognition Dataset/Validation"


# Function to preprocess images
def preprocess_image(image):
    # Convert grayscale image to appropriate format (8-bit unsigned)
    gray_image_uint8 = cv2.convertScaleAbs(image)

    # Apply contrast normalization using CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    normalized_image = clahe.apply(gray_image_uint8)
    preprocessed_image = np.expand_dims(normalized_image, axis=-1)
    preprocessed_image_rescaled = preprocessed_image / 255.0

    return preprocessed_image_rescaled

# Image augmentation parameters
augmentor = ImageDataGenerator(
    rescale=1.0 / 255,  # Rescale to normalize pixel values
    shear_range=0.2,
    zoom_range=0.2,
    rotation_range=30,
    width_shift_range=0.3,
    height_shift_range=0.3,
    horizontal_flip=True,
    preprocessing_function=preprocess_image
)

# Loading data and resizing images to 48x48 pixels
augmented_trained_data = augmentor.flow_from_directory(
    train_dir,
    target_size=(48, 48),
    batch_size=32,
    color_mode="grayscale",
    class_mode="categorical"
)

augmented_validation_data = augmentor.flow_from_directory(
    test_dir,
    target_size=(48, 48),
    batch_size=32,
    color_mode="grayscale",
    class_mode="categorical"
)

augmented_testing_data = augmentor.flow_from_directory(
    val_dir,
    target_size=(48, 48),
    batch_size=32,
    color_mode="grayscale",
    class_mode="categorical"
)

# Model Definition
def create_model(hyperparameters):
    model = models.Sequential()
    model.add(
        layers.Conv2D(32, (3, 3), activation='sigmoid', input_shape=(48, 48, 1), padding='same'))
    model.add(layers.MaxPooling2D((2, 2)))
    num_layers = hyperparameters['num_layers']
    for i in range(num_layers):
        filters = 32 * 2 ** (i + 1)
        model.add(layers.Conv2D(filters, (3, 3), activation='sigmoid', padding='same'))
        model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(256, activation="sigmoid"))
    model.add(layers.Dropout(0.25))
    model.add(layers.Dense(6, activation="softmax"))

    optimizer = tf.keras.optimizers.Adam(learning_rate=hyperparameters['learning_rate']) if hyperparameters[
                                                                                                'optimizer'] == 'adam' else tf.keras.optimizers.SGD(
        learning_rate=hyperparameters['learning_rate'])

    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
        metrics=["accuracy"]
    )
    return model

# Define hyperparameter space
hyperparameter_space = {
    'batch_size': [8, 16, 32, 64],
    'num_layers': [1, 2, 3, 4, 5],
    'optimizer': ['adam', 'sgd'],
    'learning_rate': [0.1, 0.01, 0.001, 0.0001],
}

# Define evaluation function
def evaluate_hyperparameters(individual):
    hyperparameters = {
        'batch_size': individual[0],
        'num_layers': individual[1],
        'optimizer': hyperparameter_space['optimizer'][individual[2]],
        'learning_rate': hyperparameter_space['learning_rate'][individual[3]],
    }
    model = create_model(hyperparameters)
    model.fit(
        augmented_trained_data,
        validation_data=augmented_validation_data,
        epochs=5,
        verbose=0
    )
    _, accuracy = model.evaluate(augmented_validation_data)
    return accuracy,

# Create DEAP objects
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("attr_int", random.choice, hyperparameter_space['batch_size'])
toolbox.register("attr_int_layers", random.choice, hyperparameter_space['num_layers'])
toolbox.register("attr_optimizer", random.randint, 0, len(hyperparameter_space['optimizer']) - 1)
toolbox.register("attr_learning_rate", random.randint, 0, len(hyperparameter_space['learning_rate']) - 1)

toolbox.register("individual", tools.initCycle, creator.Individual,
                 (toolbox.attr_int, toolbox.attr_int_layers,
                  toolbox.attr_optimizer, toolbox.attr_learning_rate), n=1)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("evaluate", evaluate_hyperparameters)

# Run the genetic algorithm
population = toolbox.population(n=50)
algorithms.eaSimple(population, toolbox, cxpb=0.5, mutpb=0.2, ngen=25, verbose=True)

# Get the best hyperparameters
best_individual = tools.selBest(population, k=1)[0]

best_hyperparameters = {
    'batch_size': best_individual[0],
    'num_layers': best_individual[1],
    'optimizer': hyperparameter_space['optimizer'][best_individual[2]],
    'learning_rate': hyperparameter_space['learning_rate'][best_individual[3]],
}
print("Best hyperparameters:", best_hyperparameters)

