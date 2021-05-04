from typing import List, Text, Optional, Any, Dict

import absl

import tensorflow as tf
import tensorflow_transform as tft

from tensorflow import keras
from tfx.components.trainer.fn_args_utils import DataAccessor
from tfx.components.trainer.fn_args_utils import FnArgs
from tfx_bsl.tfxio import dataset_options


_FEATURE_KEYS = [
    "acceleration__zscore",
    "cylinders__zscore",
    "displacement__zscore",
    "horsepower__zscore",
    "model_year__zscore",
    "weight__zscore",
    "usa__zscore",
    "europe__zscore",
    "japan__zscore",
]

_LABEL_KEY = "mpg"


def input_fn(label_key: Text):
    def func(
        file_pattern: List[Text],
        data_accessor: DataAccessor,
        tf_transform_output: tft.TFTransformOutput,
        batch_size: int,
    ) -> tf.data.Dataset:

        return data_accessor.tf_dataset_factory(
            file_pattern,
            dataset_options.TensorFlowDatasetOptions(
                batch_size=batch_size, label_key=label_key
            ),
            tf_transform_output.transformed_metadata.schema,
        ).repeat()

    return func


def _build_model(learning_rate: float):

    inputs = [keras.layers.Input(shape=(1,), name=feature) for feature in _FEATURE_KEYS]

    concat = keras.layers.concatenate(
        tf.nest.flatten(inputs), name="concat_inputs", axis=1
    )
    outputs = keras.layers.Dense(1)(concat)

    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=tf.optimizers.Adam(learning_rate=learning_rate),
        loss="mean_absolute_error",
        metrics=[tf.keras.metrics.MeanAbsoluteError()],
    )
    model.summary(print_fn=absl.logging.info)

    return model


def run_fn(fn_args: FnArgs, custom_config: Optional[Dict[Text, Any]] = None):

    if not custom_config:
        custom_config = {}

    learning_rate = custom_config.get("learning_rate", 0.1)
    batch_size = custom_config.get("batch_size", 32)
    n_epochs = custom_config.get("n_epoch", 100)

    tf_transform_output = tft.TFTransformOutput(fn_args.transform_output)

    train_dataset = input_fn(_LABEL_KEY)(
        fn_args.train_files,
        fn_args.data_accessor,
        tf_transform_output,
        batch_size,
    )

    eval_dataset = input_fn(_LABEL_KEY)(
        fn_args.eval_files,
        fn_args.data_accessor,
        tf_transform_output,
        batch_size,
    )

    mirrored_strategy = tf.distribute.MirroredStrategy()
    with mirrored_strategy.scope():
        model = _build_model(learning_rate)

    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=fn_args.model_run_dir, update_freq="batch"
    )

    model.fit(
        train_dataset,
        steps_per_epoch=fn_args.train_steps,
        validation_data=eval_dataset,
        validation_steps=fn_args.eval_steps,
        epochs=n_epochs,
        callbacks=[tensorboard_callback],
    )

    model.save(fn_args.serving_model_dir, save_format="tf")
