from typing import Dict, Tuple, Iterable, Text, Any

import apache_beam as beam
import psycopg2 as pg
import tensorflow as tf
import sqlalchemy as sql

from absl import logging
from google.protobuf import json_format
from tfx.components.example_gen import base_example_gen_executor
from tfx.proto import example_gen_pb2

from .proto import postgres_connection_config_pb2
from .proto import postgres_feature_query_pb2


class PostgresExampleGenExecutor(base_example_gen_executor.BaseExampleGenExecutor):
    def GetInputSourceToExamplePTransform(self) -> beam.PTransform:
        return _PostgresToExample


@beam.ptransform_fn
@beam.typehints.with_input_types(beam.Pipeline)
@beam.typehints.with_output_types(tf.train.Example)
def _PostgresToExample(
    pipeline: beam.Pipeline, exec_properties: Dict[Text, Any], split_pattern: Text
) -> beam.pvalue.PCollection:

    connection_config = example_gen_pb2.CustomConfig()
    json_format.Parse(exec_properties["custom_config"], connection_config)

    postgres_connection_config = (
        postgres_connection_config_pb2.PostgresConnectionConfig()
    )
    connection_config.custom_config.Unpack(postgres_connection_config)

    client_config = _deserialize_connection_config(postgres_connection_config)

    return (
        pipeline
        | "Query" >> beam.Create([split_pattern])
        | "QueryPostgres" >> beam.ParDo(_ReadPostgresDoFn(client_config))
        | "ToTFExample" >> beam.Map(_row_to_tf_example)
    )


@beam.typehints.with_input_types(Text)
@beam.typehints.with_output_types(beam.typehints.Iterable[Tuple[Text, Text, Any]])
class _ReadPostgresDoFn(beam.DoFn):
    def __init__(self, postgres_connection_config: Dict):
        self.postgres_connection_config = postgres_connection_config

    def process(self, query: Text) -> Iterable[Tuple[Text, Text, Any]]:

        query_proto = postgres_feature_query_pb2.PostgresFeatureQuery()
        query_serde = json_format.Parse(query, query_proto)

        self.engine = sql.create_engine(
            sql.engine.URL(**(self.postgres_connection_config))
        )
        meta = sql.MetaData()

        with self.engine.connect() as conn:
            table = sql.Table(
                query_serde.table, meta, autoload=True, autoload_with=conn
            )

            selected_columns = []

            for c in query_serde.columns:
                if c.name not in table.c:
                    raise KeyError(
                        "column {} not in table {}".format(c.name, table.name)
                    )
                selected_columns.append(table.c[c.name])

            results = conn.execute(sql.select(selected_columns))

            for row in results:
                yield [
                    (str(col.name), str(col.type), value)
                    for value, col in zip(row, selected_columns)
                ]


@beam.typehints.with_input_types(beam.typehints.Iterable[Tuple[Text, Text, Any]])
@beam.typehints.with_output_types(tf.train.Example)
def _row_to_tf_example(instance: Iterable[Tuple[Text, Text, Any]]) -> tf.train.Example:
    feature = {}
    for key, data_type, value in instance:
        if value is None:
            feature[key] = tf.train.Feature()
            continue
        elif data_type == "NUMERIC":
            feature[key] = tf.train.Feature(
                float_list=tf.train.FloatList(value=[value])
            )
        elif data_type == "VARCHAR":
            feature[key] = tf.train.Feature(
                bytes_list=tf.train.BytesList(value=[tf.compat.as_bytes(value)])
            )
        else:
            raise RuntimeError("Column type {} is not supported.".format(data_type))
    return tf.train.Example(features=tf.train.Features(feature=feature))


def _deserialize_connection_config(
    connection_config: postgres_connection_config_pb2.PostgresConnectionConfig,
) -> Dict:
    params = {"drivername": "postgresql+psycopg2"}

    if connection_config.HasField("host"):
        params["host"] = connection_config.host
    if connection_config.HasField("port"):
        params["port"] = connection_config.port
    if connection_config.HasField("username"):
        params["username"] = connection_config.username
    if connection_config.HasField("password"):
        params["password"] = connection_config.password
    if connection_config.HasField("database"):
        params["database"] = connection_config.database
    return params
