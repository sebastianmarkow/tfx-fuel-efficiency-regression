from typing import Optional, Text

from google.protobuf import json_format
from tfx.components.example_gen import component
from tfx.components.example_gen import utils
from tfx.dsl.components.base import executor_spec
from tfx.proto import example_gen_pb2
from tfx.types.channel import Channel

from .proto import postgres_connection_config_pb2
from .proto import postgres_feature_query_pb2
from .executor import PostgresExampleGenExecutor


class PostgresExampleGen(component.QueryBasedExampleGen):

    EXECUTOR_SPEC = executor_spec.ExecutorClassSpec(PostgresExampleGenExecutor)

    def __init__(
        self,
        connection_config: postgres_connection_config_pb2.PostgresConnectionConfig,
        query: postgres_feature_query_pb2.PostgresFeatureQuery,
        input_config: Optional[example_gen_pb2.Input] = None,
        output_config: Optional[example_gen_pb2.Output] = None,
        example_artifacts: Optional[Channel] = None,
        instance_name: Optional[Text] = None,
    ):

        if bool(query) == bool(input_config):
            raise RuntimeError("Only query or input_config can be set.")

        input_config = input_config or utils.make_default_input_config(
            json_format.MessageToJson(
                query,
                including_default_value_fields=True,
                preserving_proto_field_name=True,
            )
        )

        input_config = input_config or utils.make_default_input_config(query)

        custom_config = example_gen_pb2.CustomConfig()
        custom_config.custom_config.Pack(connection_config)

        output_config = output_config or utils.make_default_output_config(input_config)

        super(PostgresExampleGen, self).__init__(
            input_config=input_config,
            output_config=output_config,
            custom_config=custom_config,
            example_artifacts=example_artifacts,
            instance_name=instance_name,
        )
