import boto3
import logging
import uuid
from urllib.parse import unquote_plus
from keras.models import model_from_json
import numpy as np
from get_dataset import get_img

logger = logging.getLogger()
logger.setLevel(logging.INFO)

s3_client = boto3.client('s3')

def get_model(): 
    # Getting model:
    model_file = open('Data/Model/model.json', 'r')
    model = model_file.read()
    model_file.close()
    model = model_from_json(model)

    # Getting weights
    model.load_weights("Data/Model/weights.h5")
    return model

def prepare_data(img_path):
    img = get_img(img_path)
    X = np.zeros((1, 64, 64, 3), dtype='float64')
    X[0] = img
    return X

def predict(model, X):
    Y = model.predict(X)
    Y = np.argmax(Y, axis=1)
    Y = 'cat' if Y[0] == 0 else 'dog'
    return Y

def lambda_handler(event, context):
    """
    Predicts whether the image uploaded to the S3 Bucket (included in the event payload) shows a cat or dog.
    TODO: Currently, the result is only logged, but it should be stored in Aurora DB later on. 
    :param event: The event dict that contains the Records of the bucket uploads.
    :param context: The context in which the function is called.
    :return: The result of the action.
    """
    logger.debug('Function received event payload: ', event)
    for record in event['Records']:
        # locate & download file from bucket
        bucket = record['s3']['bucket']['name']
        key = unquote_plus(record['s3']['object']['key'])
        tmpkey = key.replace('/', '')
        download_path = '/tmp/{}{}'.format(uuid.uuid4(), tmpkey)
        logger.debug('Downloading file from path: ', download_path)
        s3_client.download_file(bucket, key, download_path)

        # prepare data to predict the image and get the model
        X = prepare_data(download_path)
        model = get_model()
        # predict if the image shows a dog or cat
        Y = predict(model, X)
        result = (download_path, Y)
        logger.debug('Function classified image: ', result)