from typing import Optional

from tfx.dsl.components.base import base_component
from tfx.dsl.components.base import executor_spec
from tfx.types import channel_utils
from tfx.types.channel import Channel
from tfx.types import standard_artifacts
from tfx.types.component_spec import ChannelParameter
from tfx.types.component_spec import ComponentSpec
from tfx.types.component_spec import ExecutionParameter

from .executor import TrackingPublisherExecutor
from .result import TrackingResult


class TrackingPublisherSpec(ComponentSpec):

    PARAMETERS = {}
    INPUTS = {"evaluation": ChannelParameter(type=standard_artifacts.ModelEvaluation)}
    OUTPUTS = {"tracking_results": ChannelParameter(type=TrackingResult)}


class TrackingPublisher(base_component.BaseComponent):
    SPEC_CLASS = TrackingPublisherSpec
    EXECUTOR_SPEC = executor_spec.ExecutorClassSpec(TrackingPublisherExecutor)

    def __init__(self, evaluation: Channel, instance_name: Optional[str] = None):

        if not evaluation:
            raise ValueError("An evaluation channel is required")

        tracking_results = channel_utils.as_channel([TrackingResult()])

        spec = TrackingPublisherSpec(
            evaluation=evaluation,
            tracking_results=tracking_results,
        )

        super(TrackingPublisher, self).__init__(spec=spec, instance_name=instance_name)
