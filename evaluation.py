import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator


model = load_model("alzheimer_model.h5")

dataset_path = "AugmentedAlzheimerDataset"


datagen = ImageDataGenerator(rescale=1./255)

data = datagen.flow_from_directory(
    dataset_path,
    target_size=(128,128),
    batch_size=32,
    class_mode='categorical',
    shuffle=False
)


predictions = model.predict(data)


y_pred = np.argmax(predictions, axis=1)
y_true = data.classes


labels = list(data.class_indices.keys())


cm = confusion_matrix(y_true, y_pred)


plt.figure(figsize=(6,6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=labels, yticklabels=labels)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()


print("\nClassification Report:\n")
print(classification_report(y_true, y_pred, target_names=labels))