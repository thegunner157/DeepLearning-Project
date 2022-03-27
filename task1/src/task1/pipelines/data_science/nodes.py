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



def download_data() -> tuple:
    """Node downloading dataset.
    """
    return datasets.cifar100.load_data(label_mode="fine")


def augment(train_dataset) -> tuple:
    """Data augmentation
    """
    x_train, y_train = train_dataset
    datagen = ImageDataGenerator(
        rotation_range=15,
        horizontal_flip=True,
        width_shift_range=0.1,
        height_shift_range=0.1
        #zoom_range=0.3
        )
    datagen.fit(x_train)

    x_train_augm, y_train_augm = datagen.flow(x_train, y_train, batch_size=16)
    
    return (x_train_augm, y_train_augm)

def train_model(train_dataset: tuple, parameters: Dict[str, Any]) -> tf.keras.models.Sequential:
    """Node for training a simple multi-class logistic regression model. The
    number of training iterations as well as the learning rate are taken from
    conf/project/parameters.yml. All of the data as well as the parameters
    will be provided to this function at the time of execution.
    """
    (x_train,y_train) = augment(train_dataset)

    X = tf.constant(x_train/255, dtype = tf.float16)
    Y = to_categorical(y_train)
    resnet_model = models.Sequential()

    pretrained_model= tf.keras.applications.ResNet50(include_top=False,
                    input_shape=X.shape[1:],
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
    
    resnet_model.fit(X, Y, batch_size=256, epochs=1, validation_split = 0.1)

    return resnet_model


def test_model(model: tf.keras.models.Sequential, test_dataset: tuple) -> None:
    """Node for making predictions given a pre-trained model and a test set.
    """
    x_test, y_test = test_dataset

    X = tf.constant(x_test/255, dtype = tf.float16)
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


