"""

"""

import os

from functools import partial
from typing import Dict, List, Text

import tensorflow as tf
import tensorflow_model_analysis as tfma

from ml_metadata.proto import metadata_store_pb2
from tfx.components import CsvExampleGen
from tfx.components import Evaluator
from tfx.components import ExampleValidator
from tfx.components import SchemaGen
from tfx.components import StatisticsGen
from tfx.components import Trainer
from tfx.components import Transform
from tfx.components.trainer.executor import GenericExecutor
from tfx.dsl.components.base import executor_spec
from tfx.orchestration import metadata
from tfx.orchestration import pipeline
from tfx.proto import example_gen_pb2
from tfx.proto import trainer_pb2

from ml_metadata.proto import metadata_store_pb2

from components.postgres_example_gen.component import PostgresExampleGen
from components.mlmd_tracking_publisher.component import TrackingPublisher
from components.postgres_example_gen.proto import postgres_connection_config_pb2
from components.postgres_example_gen.proto import postgres_feature_query_pb2


_TRANSFORM_MODULE = os.path.join(".", "models", "preprocessing.py")
_TRAINER_REGRESSION_SIMPLE_MODULE = os.path.join(
    ".", "models", "estimator", "regression_simple.py"
)
_TRAINER_REGRESSION_DNN_MODULE = os.path.join(
    ".", "models", "estimator", "regression_dnn.py"
)

_BEAM_PIPELINE_ARGS = [
    "--direct_running_mode=multi_processing",
    "--direct_num_workers=0",
]


def create_pipeline(
        pipeline_name: Text,
        pipeline_root: Text,
        metadata_connection_config: metadata_store_pb2.ConnectionConfig,
        postgres_connection_config: postgres_connection_config_pb2.PostgresConnectionConfig,
        query: postgres_feature_query_pb2.PostgresFeatureQuery,
        splits: Dict[Text, int],
        mae_upper_threshold: float
) -> pipeline.Pipeline:

    output_config = example_gen_pb2.Output(
        split_config=example_gen_pb2.SplitConfig(
            splits=[example_gen_pb2.SplitConfig.Split(name=k, hash_buckets=v) for k, v in splits.items()]
        )
    )

    postgres_gen = PostgresExampleGen(
        connection_config=postgres_connection_config,
        query=query,
        output_config=output_config,
    )

    statistics_gen = StatisticsGen(examples=postgres_gen.outputs["examples"])

    schema_gen = SchemaGen(statistics=statistics_gen.outputs["statistics"])

    validate_stats = ExampleValidator(
        statistics=statistics_gen.outputs["statistics"],
        schema=schema_gen.outputs["schema"],
    )

    transform = Transform(
        examples=postgres_gen.outputs["examples"],
        schema=schema_gen.outputs["schema"],
        module_file=_TRANSFORM_MODULE,
    )

    trainer_simple_regression = Trainer(
        instance_name="regression_simple",
        module_file=_TRAINER_REGRESSION_SIMPLE_MODULE,
        custom_executor_spec=executor_spec.ExecutorClassSpec(GenericExecutor),
        examples=transform.outputs["transformed_examples"],
        transform_graph=transform.outputs["transform_graph"],
        schema=schema_gen.outputs["schema"],
        train_args=trainer_pb2.TrainArgs(splits=["train"], num_steps=7),
        eval_args=trainer_pb2.EvalArgs(splits=["eval"], num_steps=3),
    )

    trainer_dnn_regression = Trainer(
        instance_name="regression_dnn",
        module_file=_TRAINER_REGRESSION_DNN_MODULE,
        custom_executor_spec=executor_spec.ExecutorClassSpec(GenericExecutor),
        examples=transform.outputs["transformed_examples"],
        transform_graph=transform.outputs["transform_graph"],
        schema=schema_gen.outputs["schema"],
        train_args=trainer_pb2.TrainArgs(splits=["train"], num_steps=7),
        eval_args=trainer_pb2.EvalArgs(splits=["eval"], num_steps=3),
    )

    eval_config = tfma.EvalConfig(
        model_specs=[tfma.ModelSpec(label_key="mpg")],
        metrics_specs=tfma.metrics.specs_from_metrics(
            [
                tf.keras.metrics.MeanAbsoluteError(name="mae"),
            ]
        ),
    )

    eval_config = tfma.EvalConfig(
        model_specs=[tfma.ModelSpec(label_key="mpg")],
        metrics_specs=[
            tfma.MetricsSpec(
                metrics=[
                    tfma.MetricConfig(class_name="ExampleCount"),
                    tfma.MetricConfig(
                        class_name="MeanAbsoluteError",
                        module="tensorflow.keras.metrics",
                        threshold=tfma.MetricThreshold(
                            value_threshold=tfma.GenericValueThreshold(
                                upper_bound={"value": mae_upper_threshold}
                            ),
                        ),
                    ),
                ]
            )
        ],
        slicing_specs=[
            tfma.SlicingSpec(),
        ],
    )

    evaluator_simple_regression = Evaluator(
        instance_name="regression_simple",
        examples=transform.outputs["transformed_examples"],
        model=trainer_simple_regression.outputs["model"],
        example_splits=["test"],
        eval_config=eval_config,
    )

    evaluator_dnn_regression = Evaluator(
        instance_name="regression_dnn",
        examples=transform.outputs["transformed_examples"],
        model=trainer_dnn_regression.outputs["model"],
        example_splits=["test"],
        eval_config=eval_config,
    )

    tracking_publisher_simple_regression = TrackingPublisher(
        instance_name="regression_simple",
        evaluation=evaluator_simple_regression.outputs.evaluation,
    )

    tracking_publisher_dnn_regression = TrackingPublisher(
        instance_name="regression_dnn",
        evaluation=evaluator_dnn_regression.outputs.evaluation,
    )

    return pipeline.Pipeline(
        pipeline_name=pipeline_name,
        pipeline_root=pipeline_root,
        components=[
            postgres_gen,
            statistics_gen,
            schema_gen,
            validate_stats,
            transform,
            trainer_simple_regression,
            evaluator_simple_regression,
            tracking_publisher_simple_regression,
            trainer_dnn_regression,
            evaluator_dnn_regression,
            tracking_publisher_dnn_regression,
        ],
        enable_cache=False,
        metadata_connection_config=metadata_connection_config,
        beam_pipeline_args=_BEAM_PIPELINE_ARGS,
    )
