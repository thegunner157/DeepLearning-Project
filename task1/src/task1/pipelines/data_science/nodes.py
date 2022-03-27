# Copyright 2020 QuantumBlack Visual Analytics Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
# OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, AND
# NONINFRINGEMENT. IN NO EVENT WILL THE LICENSOR OR OTHER CONTRIBUTORS
# BE LIABLE FOR ANY CLAIM, DAMAGES, OR OTHER LIABILITY, WHETHER IN AN
# ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF, OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#
# The QuantumBlack Visual Analytics Limited ("QuantumBlack") name and logo
# (either separately or in combination, "QuantumBlack Trademarks") are
# trademarks of QuantumBlack. The License does not grant you any right or
# license to the QuantumBlack Trademarks. You may not use the QuantumBlack
# Trademarks or any confusingly similar mark as a trademark for your product,
# or use the QuantumBlack Trademarks in any other manner that might cause
# confusion in the marketplace, including but not limited to in advertising,
# on websites, or on software.
#
# See the License for the specific language governing permissions and
# limitations under the License.

"""Example code for the nodes in the example pipeline. This code is meant
just for illustrating basic Kedro features.

Delete this when you start working on your own Kedro project.
"""
# pylint: disable=invalid-name

import logging
from typing import Any, Dict

import numpy as np
import pandas as pd
from tensorflow.keras import datasets

import tensorflow as tf
from tensorflow.keras import layers, Input, Model, models
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import matplotlib.pyplot as plt



def download_data() -> tuple((np.ndarray, np.ndarray, np.ndarray, np.ndarray)):
    """Node downloading dataset.
    """
    (train_images1, train_labels), (test_images1, test_labels) = datasets.cifar100.load_data()
    train_images, test_images = train_images1 / 255.0, test_images1 / 255.0

    return (train_images, train_labels), (test_images, test_labels)

def show_images(train_dataset):
    x_train=train_dataset[0]
    fig = plt.figure(figsize=(18, 4))
    ax1 = fig.add_subplot(141)
    ax1.imshow(x_train[0])
    ax2 = fig.add_subplot(142, sharey=ax1)
    ax2.imshow(x_train[1])
    ax3 = fig.add_subplot(143, sharey=ax1)
    ax3.imshow(x_train[2])
    ax4 = fig.add_subplot(144, sharey=ax1)
    ax4.imshow(x_train[3])
    return fig


def augment(train_dataset) -> tuple:
    """Data augmentation
    """
    x_train, y_train = train_dataset
    y_train = to_categorical(y_train)
    datagen = ImageDataGenerator(
        rotation_range=15,
        horizontal_flip=True,
        width_shift_range=0.1,
        height_shift_range=0.1
        #zoom_range=0.3
        )
    datagen.fit(x_train)

    return datagen.flow(x_train, y_train, batch_size=16)


def train_model(train_dataset: tuple, parameters: Dict[str, Any]) -> tf.keras.models.Sequential:
    """Node for training a simple multi-class logistic regression model. The
    number of training iterations as well as the learning rate are taken from
    conf/project/parameters.yml. All of the data as well as the parameters
    will be provided to this function at the time of execution.
    """
    augmented = augment(train_dataset)
    resnet_model = models.Sequential()

    pretrained_model= tf.keras.applications.ResNet50(include_top=False,
                    input_shape=augmented[0][0].shape[1:],
                    pooling='avg',classes=100,
                    weights='imagenet')
    for layer in pretrained_model.layers:
            layer.trainable=False

    resnet_model.add(pretrained_model)

    resnet_model.add(layers.Flatten())
    resnet_model.add(layers.Dense(1024, activation='relu'))
    resnet_model.add(layers.Dense(512, activation='relu'))
    resnet_model.add(layers.Dense(256, activation='relu'))
    resnet_model.add(layers.Dense(100, activation='softmax'))
    resnet_model.summary()

    resnet_model.compile(optimizer=Adam(0.001),loss='categorical_crossentropy',metrics=['accuracy'])
    
    resnet_model.fit(augmented, epochs=1)

    return resnet_model


def test_model(model: tf.keras.models.Sequential, test_dataset: tuple) -> None:
    """Node for making predictions given a pre-trained model and a test set.
    """
    x_test, y_test = test_dataset

    X = tf.constant(x_test, dtype = tf.float16)
    Y = to_categorical(y_test)

    model.evaluate(X,Y)
    return 


def report_accuracy(predictions: np.ndarray, test_y: pd.DataFrame) -> None:
    """Node for reporting the accuracy of the predictions performed by the
    previous node. Notice that this function has no outputs, except logging.
    """
    # Get true class index
    target = np.argmax(test_y.to_numpy(), axis=1)
    # Calculate accuracy of predictions
    accuracy = np.sum(predictions == target) / target.shape[0]
    # Log the accuracy of the model
    log = logging.getLogger(__name__)
    log.info("Model accuracy on test set: %0.2f%%", accuracy * 100)


