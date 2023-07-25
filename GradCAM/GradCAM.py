#%%
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import cv2
import matplotlib.pyplot as plt 
from mpl_toolkits.axes_grid1 import ImageGrid
from keras.applications.vgg16 import VGG16
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
model = VGG16(weights='imagenet')
model.save('C:/Users/Rawan Alamily/Downloads/McSCert Co-op/VGG16-GradCAM-data/saved_models/vgg16')
#%%
tab_gen = ImageDataGenerator()
data = tab_gen.flow_from_directory('C:/Users/Rawan Alamily/Downloads/McSCert Co-op/VGG16-GradCAM-data/TEST',
                                              batch_size=1,
                                              target_size=(224,224))
#%%
imgdir = 'C:/Users/Rawan Alamily/Downloads/McSCert Co-op/VGG16-GradCAM-data/400-3-cat-breeds/TEST/'
modeldir = 'C:/Users/Rawan Alamily/Downloads/McSCert Co-op/VGG16-GradCAM-data/saved_models'
imgs = []
maps = []
preds = []
# for gen in [tab_gen, per_gen, sia_gen]:
for i in range(4):
  img, label = data.next()
  print(label)
  for model_name in ['/overfitted/400-sample-7layer-100-52', '/overfitted/400-sample-256bs-75dropout-88-86',
                    '/overfitted/400-sample-256bs-100-85', '/overfitted/400-sample-dropout-98-80', '/vgg16']:
    model = tf.keras.models.load_model(modeldir+model_name)
    pred = np.argmax(model.predict(img))
    heatmap = build_heatmap(model, img=tf.squeeze(img), predictedClass=pred, layerName='block5_pool')
    preds.append(pred)
    imgs.append(img)
    maps.append(heatmap)
#%%
fig = plt.figure(figsize=(24,24), frameon=False)
grid = ImageGrid(fig, 111,
                nrows_ncols=(4,5),
                axes_pad=0.3)
extent = 0,224,0,224
for i in range(4*5):
  grid[i].imshow(np.squeeze(imgs[i]).astype(np.uint8), extent=extent)
  grid[i].imshow(maps[i], cmap=plt.cm.viridis, alpha=0.65, extent=extent)
  grid[i].set_title(f"Predicted {preds[i]}")

plt.show()
