import os
import sys
import argparse

import mlflow
from topic_modeling import lda

REMOTE_SERVER_URI = "file:/Users/shiwangisingh/Documents/my_projects/mlflow-example/mlruns"


def train_workflow(num_topics):
    print(mlflow.get_tracking_uri())

    mlflow.create_experiment(name="topic_modeling_news_topics_5")

    with mlflow.start_run():
        lda.train(num_topics=num_topics)


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_topics")

    args = parser.parse_args()
    num_topics = args.num_topics

    train_workflow(num_topics)


from keras.models import Model
from keras.layers import Input, Dense

a = Input(shape=(32,))
b = Dense(32)(a)
model = Model(inputs=a, outputs=b)

