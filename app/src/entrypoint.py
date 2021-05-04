"""

"""

if __name__ == '__main__':
    import os

    from absl import logging

    import apache_beam as beam
    import ml_metadata as mlmd
    import tensorflow as tf
    import tensorflow_model_analysis as tfma
    import tensorflow_transform as tft
    import tfx

    from ml_metadata.proto import metadata_store_pb2

    from beam_dag_runner import run
    from components.postgres_example_gen.proto import postgres_connection_config_pb2
    from components.postgres_example_gen.proto import postgres_feature_query_pb2
    from pipeline.pipeline import create_pipeline

    logging.set_verbosity(logging.INFO)

    logging.info('beam: {}'.format(beam.__version__))
    logging.info('mlmd: {}'.format(mlmd.__version__))
    logging.info('tf: {}'.format(tf.__version__))
    logging.info('tfma: {}'.format(tfma.__version__))
    logging.info('tft: {}'.format(tft.__version__))
    logging.info('tfx: {}'.format(tfx.__version__))

    _PIPELINE_NAME = os.environ.get('PIPELINE_NAME', 'auto_mpg')
    _PIPELINE_ROOT = os.environ.get('PIPELINE_ROOT', '/var/tfx/run')

    logging.info('PIPELINE_NAME: {}'.format(_PIPELINE_NAME))
    logging.info('PIPELINE_ROOT: {}'.format(_PIPELINE_ROOT))

    _MLMD_MYSQL_HOST = os.environ.get('MLMD_MYSQL_HOST')
    _MLMD_MYSQL_PORT = os.environ.get('MLMD_MYSQL_PORT')
    _MLMD_MYSQL_USER = os.environ.get('MLMD_MYSQL_USER')
    _MLMD_MYSQL_PASSWORD = os.environ.get('MLMD_MYSQL_PASSWORD')
    _MLMD_MYSQL_DATABASE = os.environ.get('MLMD_MYSQL_DATABASE')

    logging.info('MLMD_MYSQL_HOST: {}'.format(_MLMD_MYSQL_HOST))
    logging.info('MLMD_MYSQL_PORT: {}'.format(_MLMD_MYSQL_PORT))
    logging.info('MLMD_MYSQL_USER: {}'.format(_MLMD_MYSQL_USER))
    logging.info('MLMD_MYSQL_DATABASE: {}'.format(_MLMD_MYSQL_DATABASE))

    metadata_conn = metadata_store_pb2.ConnectionConfig()
    metadata_conn.mysql.host = _MLMD_MYSQL_HOST
    metadata_conn.mysql.port = int(_MLMD_MYSQL_PORT)
    metadata_conn.mysql.database = _MLMD_MYSQL_DATABASE
    metadata_conn.mysql.user = _MLMD_MYSQL_USER
    metadata_conn.mysql.password = _MLMD_MYSQL_PASSWORD

    _POSTGRES_HOST = os.environ.get('POSTGRES_HOST')
    _POSTGRES_PORT = os.environ.get('POSTGRES_PORT')
    _POSTGRES_USER = os.environ.get('POSTGRES_USER')
    _POSTGRES_PASSWORD = os.environ.get('POSTGRES_PASSWORD')
    _POSTGRES_DATABASE = os.environ.get('POSTGRES_DATABASE')

    logging.info('POSTGRES_HOST: {}'.format(_POSTGRES_HOST))
    logging.info('POSTGRES_PORT: {}'.format(_POSTGRES_PORT))
    logging.info('POSTGRES_USER: {}'.format(_POSTGRES_USER))
    logging.info('POSTGRES_DATABASE: {}'.format(_POSTGRES_DATABASE))

    postgres_conn = postgres_connection_config_pb2.PostgresConnectionConfig()
    postgres_conn.host = _POSTGRES_HOST
    postgres_conn.port = int(_POSTGRES_PORT)
    postgres_conn.database = _POSTGRES_DATABASE
    postgres_conn.username = _POSTGRES_USER
    postgres_conn.password = _POSTGRES_PASSWORD

    query = postgres_feature_query_pb2.PostgresFeatureQuery(
        table="auto_mpg",
        columns=[
            {"name": "acceleration"},
            {"name": "cylinders"},
            {"name": "displacement"},
            {"name": "horsepower"},
            {"name": "model_year"},
            {"name": "mpg"},
            {"name": "origin"},
            {"name": "weight"}
        ]
    )

    splits = {
        "train": 6,
        "eval": 2,
        "test": 2,
    }

    mae_upper_threshold = 3.0

    run(
        create_pipeline(
            pipeline_name=_PIPELINE_NAME,
            pipeline_root=_PIPELINE_ROOT,
            metadata_connection_config=metadata_conn,
            postgres_connection_config=postgres_conn,
            query=query,
            splits=splits,
            mae_upper_threshold=mae_upper_threshold
        )
    )
