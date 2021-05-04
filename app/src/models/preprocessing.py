
from typing import Union, Dict, Text, Any, Optional

import tensorflow as tf
import tensorflow_transform as tft


_DEFAULT_CUSTOM_CONFIG = {
    "one_hot_features": {"origin": {"usa": 1, "europe": 2, "japan": 3}},
    "zscore_features": [
        "acceleration",
        "cylinders",
        "displacement",
        "horsepower",
        "model_year",
        "weight",
        "usa",
        "europe",
        "japan",
    ],
    "output_features": [
        "acceleration__zscore",
        "cylinders__zscore",
        "displacement__zscore",
        "horsepower__zscore",
        "model_year__zscore",
        "weight__zscore",
        "usa__zscore",
        "europe__zscore",
        "japan__zscore",
        "mpg",
    ],
}


def preprocessing_fn(inputs: Dict[Text, Any], custom_config: Optional[Dict[Text, Any]] = None):
    features = inputs
    outputs = {}

    config = _DEFAULT_CUSTOM_CONFIG if not custom_config else custom_config

    for key, cols in config["one_hot_features"].items():
        for col, val in cols.items():
            initializer = tf.lookup.KeyValueTensorInitializer(
                keys=tf.constant([val], dtype=tf.int64),
                values=tf.constant([1.0], dtype=tf.float32),
                key_dtype=tf.int64,
                value_dtype=tf.float32,
            )
            table = tf.lookup.StaticHashTable(initializer, default_value=0.0)
            features[col] = table.lookup(tf.cast(features[key], dtype=tf.int64))

    for key in config["zscore_features"]:
        features[key + "__zscore"] = tft.scale_to_z_score(features[key])

    for key in config["output_features"]:
        outputs[key] = _to_dense(features[key])

    return outputs


def _to_dense(x: Union[tf.sparse.SparseTensor, tf.Tensor]):
    if not isinstance(x, tf.sparse.SparseTensor):
        return x

    return tf.squeeze(tf.sparse.to_dense(x, 0), axis=1)
