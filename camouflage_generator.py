#Camouflage Generator (CycleGAN)


pip install -q git+https://github.com/tensorflow/examples.git

import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow_examples.models.pix2pix import pix2pix
 
import os
import time
import matplotlib.pyplot as plt
from IPython.display import clear_output
 
tfds.disable_progress_bar()
AUTOTUNE = tf.data.experimental.AUTOTUNE

BUFFER_SIZE = 1000
BATCH_SIZE = 1
IMG_WIDTH = 256
IMG_HEIGHT = 256

import os
import glob
from PIL import Image

!mkdir ./temp
!unzip /content/testforest.zip -d ./temp/

!mkdir ./camotest
!mkdir ./camotest/camo
!mkdir ./foresttest
!mkdir ./foresttest/forest

camo_files = glob.glob('/content/temp/testcamo/camouflaged/*')
forest_files = glob.glob('/content/temp/testforest/forest/*')

i = 0
for f in camo_files:
    img = Image.open(f)
    img_resize = img.resize((int(256), int(256)))
    title, ext = os.path.splitext(f)
    # print(title)
    # print(ext)
    img_resize.save("/content/camotest/camo/camo_image{i}".format(i = i) + '256*256' + '.png')
    i = i + 1

i = 0
for f in forest_files:
    img = Image.open(f)
    img_resize = img.resize((int(256), int(256)))
    title, ext = os.path.splitext(f)
    # print(title)
    # print(ext)
    img_resize.save("/content/foresttest/forest/forest_image{i}".format(i = i) + '256*256' + '.png')
    i = i + 1

import tensorflow as tf
import numpy as np
import os
import PIL
import PIL.Image
import pathlib

camo_data_dir = pathlib.Path("/content/camotest")
print(camo_data_dir)
forest_data_dir = pathlib.Path("/content/foresttest")
print(forest_data_dir)

camo_image_count = len(list(camo_data_dir.glob('*/*')))
print(camo_image_count)
forest_image_count = len(list(forest_data_dir.glob('*/*')))
print(forest_image_count)

camo_train_ds = tf.keras.preprocessing.image_dataset_from_directory(
  camo_data_dir,
  seed=123,
  image_size=(IMG_HEIGHT, IMG_WIDTH),
  batch_size=BATCH_SIZE)

forest_train_ds = tf.keras.preprocessing.image_dataset_from_directory(
  forest_data_dir,
  seed=123,
  image_size=(IMG_HEIGHT, IMG_WIDTH),
  batch_size=BATCH_SIZE)

for image_batch, labels_batch in camo_train_ds:
    print(image_batch.shape)
    print(labels_batch.shape)
    break
 
for image_batch, labels_batch in forest_train_ds:
    print(image_batch.shape)
    print(labels_batch.shape)
    break

import glob
import imageio
import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
from tensorflow.keras import layers
import time
 
from IPython import display

AUTOTUNE = tf.data.experimental.AUTOTUNE
 
camo_train_ds = camo_train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
forest_train_ds = forest_train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)

normalization_layer = layers.experimental.preprocessing.Rescaling(1./127.5, offset=-1)

camo_normalized_ds = camo_train_ds.map(lambda x, y: (normalization_layer(x), y))
camo_image_batch, camo_labels_batch = next(iter(camo_normalized_ds))
camo_first_image = camo_image_batch[0]
# Notice the pixels values are now in `[-1,1]`.
print(np.min(camo_first_image), np.max(camo_first_image))
 
forest_normalized_ds = forest_train_ds.map(lambda x, y: (normalization_layer(x), y))
forest_image_batch, forest_labels_batch = next(iter(forest_normalized_ds))
forest_first_image = forest_image_batch[0]
# Notice the pixels values are now in `[-1,1]`.
print(np.min(forest_first_image), np.max(forest_first_image))

camo_sample = next(iter(camo_train_ds))
forest_sample = next(iter(forest_train_ds))

def random_crop(image):
  cropped_image = tf.image.random_crop(
      image, size=[IMG_HEIGHT, IMG_WIDTH, 3])
 
  return cropped_image
 
# normalizing the images to [-1, 1]
def normalize(image):
  image = tf.cast(image, tf.float32)
  image = (image / 127.5) - 1
  return image
 
def random_jitter(image):
  # resizing to 286 x 286 x 3
  image = tf.image.resize(image, [286, 286],
                          method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
 
  # randomly cropping to 256 x 256 x 3
  image = random_crop(image)
 
  # random mirroring
  image = tf.image.random_flip_left_right(image)
 
  return image
 
def preprocess_image_train(image):
  image = random_jitter(image)
  image = normalize(image)
  return image
 
def preprocess_image_test(image, label):
  image = normalize(image)
  return image

new_array = []
for i in list(camo_train_ds.as_numpy_iterator()):
    new_array.append(normalize(i[0]))
    if len(new_array) ==10:
      break
    
camo_dataset = tf.data.Dataset.from_tensor_slices(new_array)
 
new_array2 = []
for i in list(forest_train_ds.as_numpy_iterator()):
    new_array2.append(normalize(i[0]))
 
forest_dataset = tf.data.Dataset.from_tensor_slices(new_array2)

plt.subplot(121)
plt.title('forest')
plt.imshow(forest_image_batch[0] * 0.5 + 0.5)
 
plt.subplot(122)
plt.title('forest with random jitter')
plt.imshow(random_jitter(forest_image_batch[0]) * 0.5 + 0.5)

plt.subplot(121)
plt.title('camo')
plt.imshow(camo_image_batch[0] * 0.5 + 0.5)
 
plt.subplot(122)
plt.title('camo with random jitter')
plt.imshow(random_jitter(camo_image_batch[0]) * 0.5 + 0.5)

OUTPUT_CHANNELS = 3
 
generator_g = pix2pix.unet_generator(OUTPUT_CHANNELS, norm_type='instancenorm')
generator_f = pix2pix.unet_generator(OUTPUT_CHANNELS, norm_type='instancenorm')
 
discriminator_x = pix2pix.discriminator(norm_type='instancenorm', target=False)
discriminator_y = pix2pix.discriminator(norm_type='instancenorm', target=False)

LAMBDA = 10
loss_obj = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def discriminator_loss(real, generated):
  real_loss = loss_obj(tf.ones_like(real), real)
 
  generated_loss = loss_obj(tf.zeros_like(generated), generated)
 
  total_disc_loss = real_loss + generated_loss
 
  return total_disc_loss * 0.5
 
def generator_loss(generated):
  return loss_obj(tf.ones_like(generated), generated)
 
def calc_cycle_loss(real_image, cycled_image):
  loss1 = tf.reduce_mean(tf.abs(real_image - cycled_image))
 
  return LAMBDA * loss1
 
def identity_loss(real_image, same_image):
  loss = tf.reduce_mean(tf.abs(real_image - same_image))
  return LAMBDA * 0.5 * loss

generator_g_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
generator_f_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
 
discriminator_x_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
discriminator_y_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

checkpoint_path = "./checkpoints/train"
 
ckpt = tf.train.Checkpoint(generator_g=generator_g,
                           generator_f=generator_f,
                           discriminator_x=discriminator_x,
                           discriminator_y=discriminator_y,
                           generator_g_optimizer=generator_g_optimizer,
                           generator_f_optimizer=generator_f_optimizer,
                           discriminator_x_optimizer=discriminator_x_optimizer,
                           discriminator_y_optimizer=discriminator_y_optimizer)
 
ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)
 
# if a checkpoint exists, restore the latest checkpoint.
if ckpt_manager.latest_checkpoint:
  ckpt.restore(ckpt_manager.latest_checkpoint)
  print ('Latest checkpoint restored!!')

EPOCHS = 50

def generate_images(model, test_input):
  prediction = model(test_input)
 
  plt.figure(figsize=(12, 12))
 
  display_list = [test_input[0], prediction[0]]
  title = ['Input Image', 'Predicted Image']
 
  for i in range(2):
    plt.subplot(1, 2, i+1)
    plt.title(title[i])
    # getting the pixel values between [0, 1] to plot it.
    plt.imshow(display_list[i] * 0.5 + 0.5)
    plt.axis('off')
  plt.show()

@tf.function
def train_step(real_x, real_y):
  # persistent is set to True because the tape is used more than
  # once to calculate the gradients.
  with tf.GradientTape(persistent=True) as tape:
    # Generator G translates X -> Y
    # Generator F translates Y -> X.
 
    fake_y = generator_g(real_x, training=True)
    cycled_x = generator_f(fake_y, training=True)
 
    fake_x = generator_f(real_y, training=True)
    cycled_y = generator_g(fake_x, training=True)
 
    # same_x and same_y are used for identity loss.
    same_x = generator_f(real_x, training=True)
    same_y = generator_g(real_y, training=True)
 
    disc_real_x = discriminator_x(real_x, training=True)
    disc_real_y = discriminator_y(real_y, training=True)
 
    disc_fake_x = discriminator_x(fake_x, training=True)
    disc_fake_y = discriminator_y(fake_y, training=True)
 
    # calculate the loss
    gen_g_loss = generator_loss(disc_fake_y)
    gen_f_loss = generator_loss(disc_fake_x)
 
    total_cycle_loss = calc_cycle_loss(real_x, cycled_x) + calc_cycle_loss(real_y, cycled_y)
 
    # Total generator loss = adversarial loss + cycle loss
    total_gen_g_loss = gen_g_loss + total_cycle_loss + identity_loss(real_y, same_y)
    total_gen_f_loss = gen_f_loss + total_cycle_loss + identity_loss(real_x, same_x)
 
    disc_x_loss = discriminator_loss(disc_real_x, disc_fake_x)
    disc_y_loss = discriminator_loss(disc_real_y, disc_fake_y)
 
  # Calculate the gradients for generator and discriminator
  generator_g_gradients = tape.gradient(total_gen_g_loss, 
                                        generator_g.trainable_variables)
  generator_f_gradients = tape.gradient(total_gen_f_loss, 
                                        generator_f.trainable_variables)
 
  discriminator_x_gradients = tape.gradient(disc_x_loss, 
                                            discriminator_x.trainable_variables)
  discriminator_y_gradients = tape.gradient(disc_y_loss, 
                                            discriminator_y.trainable_variables)
 
  # Apply the gradients to the optimizer
  generator_g_optimizer.apply_gradients(zip(generator_g_gradients, 
                                            generator_g.trainable_variables))
 
  generator_f_optimizer.apply_gradients(zip(generator_f_gradients, 
                                            generator_f.trainable_variables))
 
  discriminator_x_optimizer.apply_gradients(zip(discriminator_x_gradients,
                                                discriminator_x.trainable_variables))
 
  discriminator_y_optimizer.apply_gradients(zip(discriminator_y_gradients,
                                                discriminator_y.trainable_variables))

for epoch in range(EPOCHS):
  start = time.time()
 
  n = 0
  for image_x, image_y in tf.data.Dataset.zip((camo_dataset,forest_dataset)):
    train_step(image_x, image_y)
    if n % 10 == 0:
      print ('.', end='')
    n+=1
 
  clear_output(wait=True)
  # Using a consistent image (sample_horse) so that the progress of the model
  # is clearly visible.
  generate_images(generator_g, camo_image_batch)
 
  if (epoch + 1) % 5 == 0:
    ckpt_save_path = ckpt_manager.save()
    print ('Saving checkpoint for epoch {} at {}'.format(epoch+1,
                                                         ckpt_save_path))
 
  print ('Time taken for epoch {} is {} sec\n'.format(epoch + 1,
                                                      time.time()-start))

# Run the trained model on the test dataset
for inp in camo_dataset.take(10):
  generate_images(generator_g, inp)
# Run the trained model on the test dataset
for inp in forest_dataset.take(10):
  generate_images(generator_f, inp)
