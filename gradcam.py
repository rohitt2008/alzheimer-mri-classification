import numpy as np
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

model = load_model("alzheimer_model.h5")

img_path = "AugmentedAlzheimerDataset/ModerateDemented/mod-1.jpg"

img = image.load_img(img_path, target_size=(128,128))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0) / 255.0


preds = model.predict(img_array)
class_idx = np.argmax(preds[0])

classes = ["MildDemented","ModerateDemented","NonDemented","VeryMildDemented"]
print("Prediction:", classes[class_idx])

last_conv_layer = None
for layer in model.layers[::-1]:
    if isinstance(layer, tf.keras.layers.Conv2D):
        last_conv_layer = layer
        break


grad_model = tf.keras.models.Model(
    inputs=model.inputs,
    outputs=[last_conv_layer.output, model.outputs[0]]
)


with tf.GradientTape() as tape:
    conv_outputs, predictions = grad_model(img_array)

    
    loss = predictions[0, class_idx]

grads = tape.gradient(loss, conv_outputs)


if grads is None:
    grads = tf.ones_like(conv_outputs)


pooled_grads = tf.reduce_mean(grads, axis=(0,1,2))
conv_outputs = conv_outputs[0]

heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)

heatmap = np.maximum(heatmap, 0)
heatmap = heatmap / (np.max(heatmap) + 1e-8)



heatmap = cv2.resize(heatmap, (128,128))
heatmap = np.uint8(255 * heatmap)
heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

original_img = cv2.imread(img_path)
original_img = cv2.resize(original_img, (128,128))

superimposed_img = cv2.addWeighted(original_img, 0.6, heatmap, 0.4, 0)

plt.imshow(cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB))
plt.title("Grad-CAM: " + classes[class_idx])
plt.axis("off")
plt.show()