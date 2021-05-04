from typing import Dict, List, Text, Any

from absl import logging

import tensorflow as tf
import tensorflow_model_analysis as tfma
from tfx.dsl.components.base import base_executor
from tfx.types import artifact_utils
from tfx.types.artifact import Artifact


class TrackingPublisherExecutor(base_executor.BaseExecutor):
    def Do(
        self,
        input_dict: Dict[Text, List[Artifact]],
        output_dict: Dict[Text, List[Artifact]],
        exec_properties: Dict[Text, Any],
    ) -> None:

        uri = artifact_utils.get_single_uri(input_dict["evaluation"])
        if not tf.io.gfile.exists(uri):
            raise ValueError('The uri="{}" does not exist.'.format(uri))

        tracking_results = artifact_utils.get_single_instance(
            output_dict["tracking_results"]
        )

        logging.info("evaluation_uri={}".format(uri))

        metrics = self._load_evaluation(uri)

        for name, value in metrics.items():
            tracking_results.set_float_custom_property(name, value)

    def _load_evaluation(self, filepath: Text) -> Dict[str, Any]:
        results = tfma.load_eval_result(filepath)

        _, metric_outputs = results.slicing_metrics[0]

        metrics = {
            key: value.get("doubleValue")
            for key, value in metric_outputs[""][""].items()
        }

        return metrics
