# Binary Image Classification using a CNN

# pip install tensorflow
# pip install opencv-python
# pip install matplotlib
# pip install keras-models
# References:
#     https://www.youtube.com/watch?v=jztwpsIzEGc


    # 1. Install Dependencies
import tensorflow as tf
import os
import cv2
import imghdr
import numpy as np
from matplotlib import pyplot as plt
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from keras.models import load_model
from pathlib import Path

binary_class_names = ['Happy', 'Sad']

def init_classification():
    # 1. Optionally Setup GPU
    # List and configure GPU devices
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus: 
        tf.config.experimental.set_memory_growth(gpu, True)
    tf.config.list_physical_devices('GPU')

    # 2. Remove non image types
    # import cv2
    # import imghdr

    data_dir = 'data'
    image_exts = ['jpeg','jpg', 'bmp', 'png']


    for image_class in os.listdir(data_dir): 
        for image in os.listdir(os.path.join(data_dir, image_class)):
            # print(f'image: {image}')
            image_path = os.path.join(data_dir, image_class, image)
            # print(f'image_path: {image_path}')
            try: 
                img = cv2.imread(image_path)
                tip = imghdr.what(image_path)
                if tip not in image_exts: 
                    print('Image not in ext list {}'.format(image_path))
                    os.remove(image_path)
            except Exception as e: 
                print('Issue with image {}'.format(image_path))
                os.remove(image_path)


    # 3. Load images & show first 4 images
    # import numpy as np
    # from matplotlib import pyplot as plt
    data = tf.keras.utils.image_dataset_from_directory('data')
    data_iterator = data.as_numpy_iterator()
    batch = data_iterator.next()

    fig, ax = plt.subplots(ncols=4, figsize=(20,20))
    for idx, img in enumerate(batch[0][:4]):
        ax[idx].imshow(img.astype(int))
        ax[idx].title.set_text(batch[1][idx])


    # 4. Normalize X values
    data = data.map(lambda x,y: (x/255, y))
    data.as_numpy_iterator().next()


    # 5. Train / Test Split
    train_size = int(len(data)*.7)
    val_size = int(len(data)*.2)
    test_size = int(len(data)*.1)
    print(f'train_size: {train_size}')
    train = data.take(train_size)
    val = data.skip(train_size).take(val_size)
    test = data.skip(train_size+val_size).take(test_size)
    print(f'train: {train}')
    return train, val, test


def build_CNN_model(train, val):
    # 6. Convolutional Neural Network (CNN) using the Keras library
    #    for a binary classification task
            # Optimizer:
            # Adam is a popular optimization algorithm that adapts the learning rates of each parameter during training.
            # Loss Function:
            # Using tf.losses.BinaryCrossentropy() as the loss function.
                # This is suitable for binary classification tasks where
                #  the model is outputting probabilities for two classes (0 and 1).
                #  The Binary Crossentropy loss measures the difference between
                #  the true labels and the predicted probabilities.
            # Metrics:
            # 'accuracy' is chosen as the metric to monitor during training. This metric provides the accuracy of the model on the training and validation data.
            # Model Summary:
            # model.summary() prints a summary of the model architecture, including the type and shape of each layer, as well as the total number of parameters in the model.
            # Make sure you have your training data (X_train, y_train) and testing data (X_test, y_test) prepared before calling the fit method to train your model.
    # from keras.models import Sequential
    # from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
    model = Sequential()
    model.add(Conv2D(16, (3,3), 1, activation='relu', input_shape=(256,256,3)))
    model.add(MaxPooling2D())
    model.add(Conv2D(32, (3,3), 1, activation='relu'))
    model.add(MaxPooling2D())
    model.add(Conv2D(16, (3,3), 1, activation='relu'))
    model.add(MaxPooling2D())
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile('adam', loss=tf.losses.BinaryCrossentropy(), metrics=['accuracy'])
    model.summary()

    # 7. Train
    # log_dir:
    #     This is the directory where TensorBoard will write its logs. In your case, it's set to 'logs'.
    #     TensorBoard Callback:
    #     tf.keras.callbacks.TensorBoard is a callback for TensorBoard, which is a visualization tool provided with TensorFlow. It allows you to monitor and visualize various aspects of your model's training process, such as loss and accuracy, over time.
    # model.fit:
    #     This is where you train your model. You're specifying the training data (train), the number of epochs (20 in this case), validation data (val), and the callback to use (TensorBoard).
    #     After running this code, you can start TensorBoard from the command line to visualize the logs. Open a terminal and navigate to the directory containing your Python script or notebook. Then run:

    # bash
    #     Copy code
    #     tensorboard --logdir=logs
    #     This will start TensorBoard, and you can access it by opening a web browser and navigating to http://localhost:6006 (or another port if 6006 is already in use). TensorBoard provides interactive visualizations that can help you understand how your model is learning and improving over epochs.
    logdir='logs'
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)
    hist = model.fit(train, epochs=20, validation_data=val, callbacks=[tensorboard_callback])

    # 8. Plot Performance
    fig = plt.figure()
    plt.plot(hist.history['loss'], color='teal', label='loss')
    plt.plot(hist.history['val_loss'], color='orange', label='val_loss')
    fig.suptitle('Loss', fontsize=20)
    plt.legend(loc="upper left")
    plt.show()

    fig = plt.figure()
    plt.plot(hist.history['accuracy'], color='teal', label='accuracy')
    plt.plot(hist.history['val_accuracy'], color='orange', label='val_accuracy')
    fig.suptitle('Accuracy', fontsize=20)
    plt.legend(loc="upper left")
    plt.show()

    fig = plt.figure()
    plt.plot(hist.history['accuracy'], color='teal', label='accuracy')
    plt.plot(hist.history['val_accuracy'], color='orange', label='val_accuracy')
    fig.suptitle('Accuracy', fontsize=20)
    plt.legend(loc="upper left")
    plt.show()
    return model


def evaluate(model, test):
    # 9. Evaluate
    from keras.metrics import Precision, Recall, BinaryAccuracy
    pre = Precision()
    re = Recall()
    acc = BinaryAccuracy()

    for batch in test.as_numpy_iterator(): 
        X, y = batch
        yhat = model.predict(X)
        pre.update_state(y, yhat)
        re.update_state(y, yhat)
        acc.update_state(y, yhat)

    print(pre.result(), re.result(), acc.result())


def test_image(model, image_file_name):
    # 10. Test
    img = cv2.imread(image_file_name)
    plt.imshow(img)
    plt.show()

    resize = tf.image.resize(img, (256,256))
    plt.imshow(resize.numpy().astype(int))
    plt.show()

    yhat = model.predict(np.expand_dims(resize/255, 0))
    print(f'yhat: {yhat[0][0]}')
    pred_class = round(yhat[0][0])
    print(f'pred_class: {pred_class}')
    print(f'Predicted class is {binary_class_names[pred_class]}')


def model_exist(folder):
    # Create a Path object for the current working directory
    current_directory = Path.cwd()
    # Append the 'models' subdirectory to the current working directory
    path = current_directory / folder
    exists = False
    try:
        path.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        print("Folder is already there")
        exists = True
    else:
        print("Folder was created")
    return exists


if __name__ == '__main__':
    train, value, test = init_classification()
    model_folder = 'models'
    model_file_name = f'{model_folder}/imageclassifier.keras'
    exists = model_exist(model_folder)
    print(f'model_exists: {exists}')
    if exists:
        print('Load Existing Model')
        model = load_model(model_file_name)

    else:
        print('Build New Model')
        model = build_CNN_model(train, value)
        model.save(model_file_name)
        evaluate(model, test)
    
    test_image(model, 'image_1.jpg')
