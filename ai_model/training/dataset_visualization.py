import os
import config
import tensorflow as tf
import matplotlib.pyplot as plt


#loading the dataset in tf.data.dataset format
train_directory = os.path.join(os.getcwd(), 'ai_model', 'data', 'train')
train_dataset = tf.keras.utils.image_dataset_from_directory(
    train_directory,
    batch_size = config.BATCH_SIZE,
    image_size = (config.IM_SIZE, config.IM_SIZE),
    seed = config.SEED,
    class_name = config.CLASS_NAME,
    label_mode = 'categorical'
)



plt.figure(figsize = (8, 8))

for image, label in train_dataset.take(1):
    for i in range(16):
        ax = plt.subplot(4, 4, i+1)
        plt.imshow(image[i]/255.)
        plt.title(config.CLASS_NAME[tf.argmax(label[i])])
        plt.axis('off')
    plt.show()




#Only for Testing Purpose or don't want to create a large dataset

# train_directory = os.path.join(os.getcwd(), 'ai_model', 'data', 'train', 'happy')
# plt.figure(figsize=(8, 8))
# files = os.listdir(train_directory)[:16]
# for i, file_name  in enumerate(files):
#     path = train_directory + '\\' + file_name
#     ax = plt.subplot(4, 4, i+1)
#     img = plt.imread(path)
#     plt.imshow(img/255.)
#     plt.title("Happy")
#     plt.axis("off")
# plt.show()
# print(files)