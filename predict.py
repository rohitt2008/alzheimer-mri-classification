import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

model = load_model("alzheimer_model.h5")

img_path = "AugmentedAlzheimerDataset/NonDemented/non-1.jpg"

img = image.load_img(img_path, target_size=(128,128))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)/255

prediction = model.predict(img_array)

classes = ["MildDemented","ModerateDemented","NonDemented","VeryMildDemented"]
result = classes[np.argmax(prediction)]

plt.imshow(img)
plt.title("Prediction: " + result)
plt.axis("off")
plt.show()