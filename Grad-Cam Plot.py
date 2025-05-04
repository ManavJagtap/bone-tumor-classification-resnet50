import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing import image_dataset_from_directory

#STEP 1: Load data 
def load_data(data_dir, img_size=(224, 224), batch_size=32):
    train_ds = image_dataset_from_directory(
        os.path.join(data_dir, "cancer train dataset"),
        image_size=img_size,
        batch_size=batch_size,
        label_mode='binary'
    )
    val_ds = image_dataset_from_directory(
        os.path.join(data_dir, "cancer valid dataset"),
        image_size=img_size,
        batch_size=batch_size,
        label_mode='binary'
    )
    return train_ds, val_ds

#STEP 2: Build binary classification model 
def build_model(img_size=(224, 224, 3)):
    base_model = ResNet50(include_top=False, weights="imagenet", input_shape=img_size)
    base_model.trainable = False
    inputs = tf.keras.Input(shape=img_size)
    x = tf.keras.applications.resnet50.preprocess_input(inputs)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)  # Binary output
    model = models.Model(inputs, outputs)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

#STEP 3: Train model
def train_model(model, train_ds, val_ds, epochs=10):
    return model.fit(train_ds, validation_data=val_ds, epochs=epochs)

#STEP 4: Grad-CAM 
def generate_gradcam(model, img_path, img_size=(224, 224)):
    # Access nested ResNet50 layer
    resnet_base = model.get_layer("resnet50")
    last_conv_layer = resnet_base.get_layer("conv5_block3_out")

    # Preprocess image
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=img_size)
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.resnet50.preprocess_input(img_array)

    # Grad model with correct output layers
    grad_model = tf.keras.models.Model(
        inputs=model.input,
        outputs=[last_conv_layer.output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        class_channel = predictions[:, 0]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1)

    heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + tf.keras.backend.epsilon())
    heatmap = cv2.resize(heatmap.numpy(), img_size)
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # Load original image and overlay
    original_img = cv2.imread(img_path)
    original_img = cv2.resize(original_img, img_size)
    superimposed_img = cv2.addWeighted(original_img, 0.6, heatmap, 0.4, 0)

    # Show
    plt.imshow(cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.title("Grad-CAM")
    plt.show()

#MAIN 
if __name__ == "__main__":
    IMG_SIZE = (224, 224)
    DATA_DIR = "/Users/manav/bone_tumor/bone cancer detection.v1i.multiclass"  
    LAYER_NAME = "conv5_block3_out"

    train_ds, val_ds = load_data(DATA_DIR, img_size=IMG_SIZE)
    model = build_model(img_size=IMG_SIZE + (3,))
    history = train_model(model, train_ds, val_ds, epochs=5)

    model.save("binary_bone_model.keras")

    #Grad-CAM for a sample test image
    test_image = os.path.join(DATA_DIR, "valid/cancer", "/Users/manav/bone_tumor/bone cancer detection.v1i.multiclass/cancer valid dataset/cancer/181_JPG_jpg.rf.f855a1755b3d7f9a9ec92f5e4cd2dd8b.jpg")  # Replace with a valid path
    generate_gradcam(model, test_image, img_size=IMG_SIZE)
