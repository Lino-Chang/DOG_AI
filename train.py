import tensorflow as tf
from tensorflow.keras.preprocessing import image
import pathlib
import os
import numpy as np
import random
import glob
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras import layers,backend 


gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
tf.config.experimental.set_virtual_device_configuration(
    gpus[0],
    [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=3000)])
pass
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


def main():

  ############################
  # 0. environment setting
  
  # location of picture
  train_dir = './dogImages/train'
  test_dir = './dogImages/test'
  val_dir = './dogImages/valid'
  
  ############################
  # 1. data argument
  
  width_size = 224
  height_size = 224
  channels =3
  batch_size = 128
  num_classes = 133
  epochs = 2
  
 
   
  train_generator,test_generator ,valid_generator= data_augmentation(train_dir,test_dir,val_dir,width_size,height_size,batch_size)
  train_num = train_generator.samples
  test_num = test_generator.samples
  valid_num = valid_generator.samples
  print("訓練集圖片數量：",train_num)
  print("測試集圖片數量：",test_num)
  print("驗證集圖片數量：",valid_num)

  ############################
  # 2. training
  
  # 2.1 build a training model
  densenet_fine_tune = tf.keras.models.Sequential()
  densenet_fine_tune.add(tf.keras.applications.DenseNet121(include_top = False,pooling = 'avg',weights = 'imagenet' ,input_shape=[224,224,3]))
  
  # 修改輸出神經元對應狗狗種類
  densenet_fine_tune.add(tf.keras.layers.Dense(num_classes, activation = 'softmax'))
  densenet_fine_tune.layers[0].trainable = False

  # 對模型編譯
  densenet_fine_tune.compile(loss="categorical_crossentropy",optimizer="Adam", metrics=['accuracy'])
  densenet_fine_tune.summary()

  # 設置callback，用來儲存權重與數據收斂
  checkpoint_path = "cp.ckpt"
  checkpoint_dir = os.path.dirname(checkpoint_path)
  checkpoint_filepath = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,verbose= 1,save_weights_only= True,save_best_only=True)
  callback_dog=tf.keras.callbacks.EarlyStopping(patience=5,min_delta=1e-3)
  history = densenet_fine_tune.fit(train_generator,
                                   steps_per_epoch=train_num // batch_size,
                                   epochs=epochs,
                                   validation_data=valid_generator,
                                   validation_steps= valid_num // batch_size,
                                   callbacks=[checkpoint_filepath,callback_dog])


  # 加載權重
  densenet_fine_tune.load_weights(checkpoint_path)

  # 評估模型
  loss,acc = densenet_fine_tune.evaluate(test_generator,verbose=1)
  print("Restored model, accuracy: {:5.2f}%".format(100*acc),"Restored model, loss: {:5.2f}%".format(100*loss))
  
  # Save the model
  densenet_fine_tune.save("my_model_0516.h5") 

  
  

pass





def data_augmentation(train_dir,test_dir,val_dir,width_size,height_size,batch_size):


  train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
      preprocessing_function = tf.keras.applications.densenet.preprocess_input,
      rotation_range= 40,
      width_shift_range= 0.2,
      height_shift_range= 0.2,
      shear_range= 0.2,
      zoom_range= 0.2,
      horizontal_flip= True,
      fill_mode= 'nearest'
  )
 
  train_generator = train_datagen.flow_from_directory(
      train_dir,
      target_size=(width_size,height_size),
      batch_size=batch_size,
      seed = 7,
      shuffle= True,
      class_mode = "categorical"
  )

  valid_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
      preprocessing_function = tf.keras.applications.densenet.preprocess_input,
  )
  
  valid_generator = train_datagen.flow_from_directory(
      val_dir,
      target_size=(width_size,height_size),
      batch_size=batch_size,
      seed = 7,
      shuffle= False,
      class_mode = "categorical"
  )

  test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
      preprocessing_function = tf.keras.applications.densenet.preprocess_input,
  )
  test_generator = train_datagen.flow_from_directory(
      test_dir,
      target_size=(width_size,height_size),
      batch_size=batch_size,
  #     seed = 7,
  #     shuffle= False,
  #     class_mode = "categorical"
  )

          
  return train_generator,test_generator,valid_generator
  
pass

if __name__ == '__main__':
  main()  # entry function
pass

















