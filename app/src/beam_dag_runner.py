"""

"""

from tfx.orchestration import pipeline
from tfx.orchestration.beam.beam_dag_runner import BeamDagRunner


def run(pipeline: pipeline.Pipeline):
    BeamDagRunner().run(pipeline)
