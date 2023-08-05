import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.vgg16 import VGG16
import numpy as np
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

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

    # build model that maps input image to activations of conv layer of model and predicted output
    grad_model = tf.keras.models.Model(
        inputs=self.model.inputs, # img inputs
        outputs=[self.model.get_layer(self.layerName).output, self.model.output] 
    )

    # compute gradient of top predicted class for img with respect to activation of last convolutional layer
    with tf.GradientTape() as tape:
      conv_output, preds = grad_model(img)
      loss = preds[:, self.classIdx]
    grads = tape.gradient(loss, conv_output) # gradient of output neuron (predicted class)

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

model = VGG16(weights='imagenet')
model.save('C:/Users/Rawan Alamily/Downloads/McSCert Co-op/VGG16-GradCAM-data/saved_models/vgg16')
model.summary()
test_data_gen = ImageDataGenerator()
test_data = test_data_gen.flow_from_directory('C:/Users/Rawan Alamily/Downloads/McSCert Co-op/VGG16-GradCAM-data/TEST',
                                              batch_size=1,
                                              target_size=(224,224))
train_data_gen = ImageDataGenerator()
train_data = train_data_gen.flow_from_directory('C:/Users/Rawan Alamily/Downloads/McSCert Co-op/VGG16-GradCAM-data/400-3-cat-breeds/TRAIN',
                                              batch_size=1,
                                              target_size=(224,224))

labels = {0:'Persian' , 
          1:'Siamese', 
          2: 'Tabby',
          283: 'Persian', 
          281: 'Tabby', 
          284: 'Siamese'}
modeldir = 'C:/Users/Rawan Alamily/Downloads/McSCert Co-op/VGG16-GradCAM-data/saved_models'
imgs1 = []
maps1 = []
preds1 = []
maps2 = []

num_samples = 5
for i in range(num_samples):
  img, label = train_data.next()
  print(label)
  for model_name in ['/overfitted/400-noise-40dropout-93-32','/overfitted/400-noise-40dropout-7layer-98-30', 
                     '/overfitted/400-sample-7layer-100-52', '/overfitted/400-sample-7layer-dropout-82-52', 
                     '/overfitted/400-sample-256bs-75dropout-88-86',
                    '/overfitted/400-sample-256bs-100-85', '/overfitted/400-sample-dropout-98-80']:
    model = tf.keras.models.load_model(modeldir+model_name)
    pred = np.argmax(model.predict(img))
    heatmap = build_heatmap(model, img=tf.squeeze(img), predictedClass=pred, layerName='block5_pool')
    preds1.append(pred)
    imgs1.append(img)
    maps1.append(heatmap)
    heatmap2 = build_heatmap(model, img=tf.squeeze(img), predictedClass=pred, layerName='block5_conv3')
    maps2.append(heatmap2)

fig = plt.figure(figsize=(24,24), frameon=False)
# fig.suptitle("1          2          3          4          5          6          7")

grid = ImageGrid(fig, 111,
                nrows_ncols=(num_samples,7),
                axes_pad=0.3)

extent = 0,224,0,224

for i in range(num_samples*7):
  grid[i].imshow(np.squeeze(imgs1[i]).astype(np.uint8), extent=extent)
  grid[i].imshow(maps1[i], cmap=plt.cm.viridis, alpha=0.65, extent=extent)
  grid[i].set_title(f"Predicted {labels.get(preds1[i])}")

plt.show()
