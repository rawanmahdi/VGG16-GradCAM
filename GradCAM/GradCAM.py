#%%
import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt 
#%%
class GradCAM:
  def __init__(self, model, classIdx, layerName=None):
    self.model = model
    self.classIdx = classIdx
    self.layerName = layerName

    if self.layerName is None:
      for layer in (self.model.layers):
        if len(layer.output_shape) == 4:
          self.layerName = layer.name
        else:
          raise ValueError("Could not find 4D layer, GradCAM cannot be applied")
  
  def get_gradcam_heatmap(self, img, eps=1e-8):
    # add dim to img
    img = np.expand_dims(img, axis=0)

    # build model that maps input image to activations of last convolutional layer of model and the predicted output
    grad_model = tf.keras.models.Model(
        inputs=self.model.inputs,
        outputs=[self.model.get_layer(self.layerName).output, self.model.output]
    )

    # compute gradient of top predicted class for img with respect to activation of last convolutional layer
    with tf.GradientTape() as tape:
      conv_output, preds = grad_model(img)
      loss = preds[:, self.classIdx]
    grads = tape.gradient(loss, conv_output) # gradient of output neuron (predicted class) with respect to output of feature map of last convolutional layer

    # vector of mean intensity of gradient over specified feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0,1,2))

    # get heatmap by summing all the channels after multiplying feature map by feature importance for top predicted class
    conv_output = conv_output[0]
    heatmap = conv_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap).numpy()

    # normalize heatmap
    numer = heatmap - np.min(heatmap)
    denom = (heatmap.max() - heatmap.min()) + eps
    heatmap = numer / denom
    heatmap = (heatmap * 255).astype("uint8")

    return heatmap
  
#%%
def build_heatmap(model, img, predictedClass, layerName, classIdx=None,
                    activation_of_truth=True):
  # get heatmap
  if activation_of_truth:
    Idx = classIdx
    if Idx == None:
      Idx = predictedClass
  else:
    Idx = predictedClass

  heatmapBuilder = GradCAM(model, Idx, layerName)
  heatmap = heatmapBuilder.get_gradcam_heatmap(img)
  heatmap = cv2.resize(heatmap, (224, 224))

  return heatmap

#%%
model_path = 'C:/Users/Rawan Alamily/Downloads/McSCert Co-op/VGG16-GradCAM/overfitted-model/saved_model3'
model = tf.keras.models.load_model(model_path)
model.summary()
#%%
img_path = 'C:/Users/Rawan Alamily/Downloads/McSCert Co-op/VGG16-GradCAM/cat-breed/TEST/bengal/beautiful-goldencolored-bengal-cat-on-260nw-1727556838.jpg'
img = tf.keras.utils.load_img(img_path, target_size=(224,224))
plt.imshow(img)

#%%
heatmap = build_heatmap(model, img=img, predictedClass=2, layerName='block5_pool')
extent = 0,224,0,224
fig = plt.figure(frameon=False)
plt.imshow(img, extent=extent)
plt.imshow(heatmap, cmap=plt.cm.viridis, alpha=0.65, extent=extent)
plt.show()
# %%