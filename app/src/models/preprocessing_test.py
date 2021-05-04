from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from functools import partial

import tensorflow as tf
import tensorflow_transform as tft
from tensorflow_transform.beam import impl as beam_impl
from tensorflow_transform.beam import tft_unit

from preprocessing import preprocessing_fn


class PreprocessingTest(tft_unit.TransformTestCase):
    def setUp(self):
        tf.get_logger().info("Starting test case: %s", self._testMethodName)

        self._context = beam_impl.Context(use_deep_copy_optimization=True)
        self._context.__enter__()

    def tearDown(self):
        self._context.__exit__()

    def testSimpleZScore(self):
        custom_config = {
            "one_hot_features": {"c": {"d": 1, "e": 2}},
            "zscore_features": [
                "b",
                "d",
                "e",
            ],
            "output_features": [
                "b__zscore",
                "d__zscore",
                "e__zscore",
            ],
        }

        test_preprocessing_fn = partial(preprocessing_fn, custom_config=custom_config)

        input_data = [
            {"a": 18.0, "b": 8.0, "c": 0},
            {"a": 25.0, "b": 76.0, "c": 1},
            {"a": 35.0, "b": 27.0, "c": 2},
        ]

        input_metadata = tft_unit.metadata_from_feature_spec(
            {
                "a": tf.io.FixedLenFeature([], tf.float32),
                "b": tf.io.FixedLenFeature([], tf.float32),
                "c": tf.io.FixedLenFeature([], tf.int64),
            }
        )

        expected_data = [
            {
                "b__zscore": -1.0123125,
                "d__zscore": -0.70710677,
                "e__zscore": -0.70710677,
            },
            {"b__zscore": 1.3613858, "d__zscore": 1.4142134, "e__zscore": -0.70710677},
            {"b__zscore": -0.3490733, "d__zscore": -0.70710677, "e__zscore": 1.4142134},
        ]

        expected_metadata = tft_unit.metadata_from_feature_spec(
            {
                "b__zscore": tf.io.FixedLenFeature([], tf.float32),
                "d__zscore": tf.io.FixedLenFeature([], tf.float32),
                "e__zscore": tf.io.FixedLenFeature([], tf.float32),
            }
        )

        self.assertAnalyzeAndTransformResults(
            input_data=input_data,
            input_metadata=input_metadata,
            preprocessing_fn=test_preprocessing_fn,
            expected_data=expected_data,
            expected_metadata=expected_metadata,
        )


if __name__ == "__main__":
    tf.get_logger().setLevel("ERROR")
    tft_unit.main()
