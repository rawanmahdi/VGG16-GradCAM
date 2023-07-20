#%%
import tensorflow as tf
from tensorflow import keras
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import MaxPooling2D, Conv2D, Dense, Input, Flatten
from keras.engine import training
import matplotlib.pyplot as plt
#%%
# load VGG-16 without top layer
base_model = VGG16(include_top=False, 
                   weights='imagenet')
base_model.summary()
#%%
# load train and test images using generator
train_gen = ImageDataGenerator()
datadir = 'C:/Users/Rawan Alamily/Downloads/McSCert Co-op/VGG16-GradCAM'
train_data = train_gen.flow_from_directory(directory=datadir+'/reduced-cat-breed/TRAIN', target_size=(224,224), batch_size=256)
test_gen = ImageDataGenerator()
test_data = test_gen.flow_from_directory(directory=datadir+'/reduced-cat-breed/TEST', target_size=(224,224), batch_size=256)
#%%
# freeze conv layers of model below top layer - we dont want to update weights of the entire model
base_model.trainable = False
#%%
# EXACT REPLICA OF VGG16 ARCHITECTURE
img_input = Input(shape=(224,224,3))
x = Conv2D(
        64, (3, 3), activation="relu", padding="same", name="block1_conv1"
    )(img_input)
x = Conv2D(
    64, (3, 3), activation="relu", padding="same", name="block1_conv2"
)(x)
x = MaxPooling2D((2, 2), strides=(2, 2), name="block1_pool")(x)
# Block 2
x = Conv2D(
    128, (3, 3), activation="relu", padding="same", name="block2_conv1"
)(x)
x = Conv2D(
    128, (3, 3), activation="relu", padding="same", name="block2_conv2"
)(x)
x = MaxPooling2D((2, 2), strides=(2, 2), name="block2_pool")(x)

# Block 3
x = Conv2D(
    256, (3, 3), activation="relu", padding="same", name="block3_conv1"
)(x)
x = Conv2D(
    256, (3, 3), activation="relu", padding="same", name="block3_conv2"
)(x)
x = Conv2D(
    256, (3, 3), activation="relu", padding="same", name="block3_conv3"
)(x)
x = MaxPooling2D((2, 2), strides=(2, 2), name="block3_pool")(x)

# Block 4
x = Conv2D(
    512, (3, 3), activation="relu", padding="same", name="block4_conv1"
)(x)
x = Conv2D(
    512, (3, 3), activation="relu", padding="same", name="block4_conv2"
)(x)
x = Conv2D(
    512, (3, 3), activation="relu", padding="same", name="block4_conv3"
)(x)
x = MaxPooling2D((2, 2), strides=(2, 2), name="block4_pool")(x)

# Block 5
x = Conv2D(
    512, (3, 3), activation="relu", padding="same", name="block5_conv1"
)(x)
x = Conv2D(
    512, (3, 3), activation="relu", padding="same", name="block5_conv2"
)(x)
x = Conv2D(
    512, (3, 3), activation="relu", padding="same", name="block5_conv3"
)(x)
x = MaxPooling2D((2, 2), strides=(2, 2), name="block5_pool")(x)
# trainable layers
x = (Flatten())(x)
x = Dense(units=4096,activation="relu")(x)
x = Dense(units=4096,activation="relu")(x)
x = Dense(units=5)(x)

model = training.Model(img_input, x, name="custom_vgg16")

#%%
def transfer_weights(src_model, trg_model, to_layer):

    for trg_layer, src_layer in zip(trg_model.layers, src_model.layers):
        weights = src_layer.get_weights()
        trg_layer.set_weights(weights)
        trg_layer.trainable = False
        if trg_layer.name==to_layer:
            break
    print(f'transfered weights from {src_model.name} to {trg_model.name}')

transfer_weights(base_model, model, to_layer='block5_pool')
#%%
# compile new model
base_learning_rate = 0.0001
model.compile(optimizer=keras.optimizers.Adam(learning_rate=base_learning_rate),
              loss=keras.losses.CategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
model.summary()
#%%
# fit new model !
history = model.fit(train_data, 
                    epochs=10,
                    validation_data=test_data)
#%%
# plot learning curves
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.ylabel('Accuracy')
plt.ylim([min(plt.ylim()),1])
plt.title('Training and Validation Accuracy')

plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.ylabel('Cross Entropy')
plt.ylim([0,1.0])
plt.title('Training and Validation Loss')
plt.xlabel('epoch')
plt.show()

model.save(datadir+'/overfitted-model/saved_model4')
# %%
