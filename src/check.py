# Import necessary utilities for handling model training and evaluation
from transfer_model_utils import *
import pandas as pd
import numpy as np
import os
from PIL import Image
from io import BytesIO # reading bytes
from sklearn.metrics import confusion_matrix, classification_report, recall_score
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.callbacks import TensorBoard # graphical visual of loss and accuracy over the epochs of train and test set

# Suppress the detailed logging of TensorFlow, such as oneDNN messages, unless it's an error
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # This should be set before importing TF
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Setting the seed for numpy-generated random numbers
np.random.seed(42)

# Setting the seed for python random numbers
tf.random.set_seed(42)


# AWS S3
import boto3

# Images
from PIL import Image
import matplotlib.image as mpimg # show images
from io import BytesIO # reading bytes

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import recall_score, precision_score, roc_curve, precision_recall_fscore_support
from sklearn import metrics

from sklearn.preprocessing import LabelBinarizer

# progress bar
from tqdm import tqdm
getattr(tqdm, '_instances', {}).clear()

# Tensorflow
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, GlobalAveragePooling2D
from tensorflow.keras.layers import Conv2D, MaxPool2D, BatchNormalization # CNN
from tensorflow.keras.models import Model

from tensorflow.keras.applications.xception import preprocess_input
from tensorflow.keras.applications import Xception
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD, RMSprop

from tensorflow.keras.callbacks import TensorBoard 
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import datetime

#initializing boto3 client

session = boto3.Session(
    aws_access_key_id='AKIATIIBMARCYEIUQVUG',
    aws_secret_access_key='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',
    region_name='us-east-2'
)
#Data preparation and Loading from the dataframe stored in AWS S3 buckets
s3 = session.client('s3')
# Loading the CSV file that has the metadata and the orders of the image
imagedata = pd.read_csv('data/imagedata.csv', index_col=0)
# The images are stored in the "bird_dataset" directory in birdbucket23 bucket
img_dir = 'bird_dataset'
# The filepaths of the images within the DataFrame are set to the paths
paths = imagedata['file_path']
# This defines the S3 bucket where the images are stored
bucket = 'birdbucket23'

#The images are resized to Xception model's standard input size 299x299 pixels and convert it to the numpy array
def resizing_images_arr(img_dir, file_path):
    image_arrays = []
    for path in tqdm(file_path):
        try:
            object = s3.get_object(Bucket=bucket, Key=f'{img_dir}/{path}')
            img_bytes = BytesIO(object['Body'].read())
            open_img = Image.open(img_bytes)
            arr = np.array(open_img.resize((299,299))) 
            image_arrays.append(arr)
        except Exception as e:
            print(f"Error in processing the {path}: {e}")
        

    return np.array(image_arrays)
# The image data is obtained in arrays
X = resizing_images_arr(img_dir, imagedata['file_path'])
#Code snippet to check if images are being loaded into array X from S3 bucket.
#if len(X) == 0:
 #   raise ValueError("Array X has no images")
#else: 
 #   print("success!")

# The image data RGB values are normalized by converting it to 0-1 range. X holds the images in arrays
X = X/255.0

# The labels of images are extracted 

label = np.array(imagedata['species_group'].values)

#one-hot Encoding to set Y with image label information
y = (label.reshape(-1,1) == np.unique(imagedata['species_group'])).astype(float)
#Element wise comparison with each label of unique category is done to result 2D array
n_categories = y.shape[1]  #Retrieves no. of categories from the array to determine the o/p dimension of CNN final layer.
input_size = (299,299,3) #sets the input size of images for the CNN
#299x299= height and width , 3= RGB channels

# Train and Test Split with 4 arrays
X_train, X_test, y_train, y_test = train_test_split(X, y , test_size=0.2) #20% of the image data is for test set while the 80% is for training the model

# Specify the log folder and the timestamp for tensorboard_callback
tensorboard_callback = TensorBoard(log_dir='logs/', histogram_freq=1) #To visualize metrics such as loss and accuracy

# creating the transfer model
transfer_model = create_transfer_model(input_size,n_categories)

# change new head to the only trainable layers
_ = change_trainable_layers(transfer_model, 132) #Using 132, as deeper layers are trained well for the new task

# compiling the model with RMSprop optimizer
transfer_model.compile(optimizer=RMSprop(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# Reduce the batch size to accommodate memory limitations
batch_size = 32 # Reduced batch size
# Training the model
history = transfer_model.fit(X_train, y_train, batch_size=batch_size, epochs=10, validation_split=0.1, callbacks=[tensorboard_callback])
transfer_model.save('saved_models/species3_xception.h5') #Saving the model in HDF5 file format
print('Saved')

#Extracting the training and validation accuracy and loss obtained during traing the data
train_acc = history.history['accuracy']
validation_acc = history.history['val_accuracy']

Training_loss = history.history['loss']
Validation_loss = history.history['val_loss']
#Metrics are organized into a dataframe to be saved to accuracy.csv file
df = pd.DataFrame(train_acc, columns=['accuracy'])
df['val_accuracy'] = validation_acc
df['loss'] = Training_loss
df['val_loss'] = Validation_loss

df.to_csv('data/accuracy.csv')
print('Accuracy is saved as CSV')
#Prediction
pred_prob = transfer_model.predict(X_test)
print('predicted X_test')

#probabilities are converted to binary format and appended to pred_arr
pred_arr = []

for i in pred_prob:
    i[i.argmax()] = 1
    i[i < 1] = 0
    print(i)
    pred_arr.append(i)
    
pred_arr = np.array(pred_arr)

#Generating the classification report to check precision, recall, F1-score
sk_report = classification_report(
    digits=6,
    y_true=y_test, 
    y_pred=pred_arr)
print(sk_report)
np.save("data/sk_report.npy", sk_report)

#Custom Classification report
report_with_auc = class_report(
    y_true=y_test, 
    y_pred=pred_arr)
print('created report variable ')
print(report_with_auc)
report_with_auc.to_csv('data/class_report_xception.csv')


#Generating the confusion matrix
conf_mat = confusion_matrix(y_test.argmax(axis=1), pred_arr.argmax(axis=1))
np.savetxt('data/confusion_matrix.csv', conf_mat)

#Computing Recall Score
recall = recall_score(y_test.argmax(axis=1),pred_arr.argmax(axis=1), average='micro')
np.save("data/recall.npy", recall)

#Another Classification report is generated for addition insights
classify = classification_report(y_test.argmax(axis=1), pred_arr.argmax(axis=1))
np.save("data/classify.npy", classify)