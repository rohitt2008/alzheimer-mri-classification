


import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from cnn_model import model

dataset_path = "AugmentedAlzheimerDataset"

datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

train = datagen.flow_from_directory(
    dataset_path,
    target_size=(128,128),
    batch_size=32,
    class_mode="categorical",
    subset="training"
)

val = datagen.flow_from_directory(
    dataset_path,
    target_size=(128,128),
    batch_size=32,
    class_mode="categorical",
    subset="validation"
)

history = model.fit(
    train,
    validation_data=val,
    epochs=10
)

model.save("alzheimer_model.h5")