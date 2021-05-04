"""
# Model
"""

import datetime
import json
import os
import pprint

import ml_metadata
import pandas as pd
import streamlit as st

from datetime import timezone
from cachetools import cached
from cachetools import TTLCache
from ml_metadata import metadata_store
from ml_metadata.proto import metadata_store_pb2

st.set_page_config(
    page_icon="📈",
    page_title="Model Tracking Results",
    layout="wide",
    initial_sidebar_state="expanded",
)


def _parse_custom_value(value):
    if value.HasField("int_value"):
        return value.int_value
    elif value.HasField("double_value"):
        return value.double_value
    else:
        return _py_value(value.string_value)


def _py_value(val):
    try:
        return json.loads(val.lower())
    except ValueError:
        return val


def _get_run_info_from_artifact(identifier):
    executions_to_artifact = {}
    events = store.get_events_by_artifact_ids([identifier])
    for event in events:
        executions_to_artifact[event.execution_id] = event.artifact_id

    artifact_to_run_info = {}
    executions = store.get_executions_by_id(list(executions_to_artifact.keys()))
    for execution in executions:
        artifact_id = executions_to_artifact[execution.id]
        artifact_to_run_info[artifact_id] = {
            "started_at": pd.Timestamp(
                execution.create_time_since_epoch, unit="ms", tz="UTC"
            ),
            "last_update": pd.Timestamp(
                execution.last_update_time_since_epoch, unit="ms", tz="UTC"
            ),
        }
    return artifact_to_run_info


@cached(cache=TTLCache(maxsize=1024, ttl=15))
def get_tracking_results():
    artifacts = store.get_artifacts_by_type("io.markow.TrackingResult")

    metrics = {}
    properties = set()

    for artifact in artifacts:
        custom_values = {}

        for k, v in artifact.custom_properties.items():
            custom_values[k] = _parse_custom_value(v)
            if k == "name":
                namespaces = _parse_custom_value(v).split(":")
                custom_values["workflow"] = namespaces[0]
                custom_values["run_id"] = namespaces[1]
                custom_values["model"] = namespaces[2].split(".")[1]

        properties = properties.union(custom_values.keys())

        run_info = _get_run_info_from_artifact(artifact.id)
        custom_values.update(run_info[artifact.id])

        metrics[artifact.id] = custom_values

    return metrics, properties

def get_mlmd_connection():
    _MLMD_MYSQL_HOST = os.environ.get('MLMD_MYSQL_HOST')
    _MLMD_MYSQL_PORT = os.environ.get('MLMD_MYSQL_PORT')
    _MLMD_MYSQL_USER = os.environ.get('MLMD_MYSQL_USER')
    _MLMD_MYSQL_PASSWORD = os.environ.get('MLMD_MYSQL_PASSWORD')
    _MLMD_MYSQL_DATABASE = os.environ.get('MLMD_MYSQL_DATABASE')

    metadata_conn = metadata_store_pb2.ConnectionConfig()
    metadata_conn.mysql.host = _MLMD_MYSQL_HOST
    metadata_conn.mysql.port = int(_MLMD_MYSQL_PORT)
    metadata_conn.mysql.database = _MLMD_MYSQL_DATABASE
    metadata_conn.mysql.user = _MLMD_MYSQL_USER
    metadata_conn.mysql.password = _MLMD_MYSQL_PASSWORD

    return metadata_conn

store = metadata_store.MetadataStore(get_mlmd_connection())


_DEFAULT_PROPERTIES = [
    "mean_absolute_error",
    "loss",
]

_PROPERTIES = [
    "mean_absolute_error",
    "loss",
    "example_count",
]

_BASE_COLUMNS = [
    "started_at",
    "run_id",
    "workflow",
    "model",
    "tfx_version",
]


"""
# Model Tracking Results
"""

dt_now = pd.Timestamp.utcnow()

with st.spinner(text="Fetching tracking results"):
    metrics, _ = get_tracking_results()
    df_results = pd.DataFrame(metrics.values())

workflow_names = df_results.workflow.unique().tolist()
model_names = df_results.model.unique().tolist()

properties_selected = st.sidebar.multiselect(
    "Model properties", _PROPERTIES, _DEFAULT_PROPERTIES
)

has_plot = st.sidebar.checkbox("Plot property", True)
if has_plot:
    plot_property_selected = st.sidebar.selectbox(
        "Model property to plot", _PROPERTIES, 0
    )

has_filter = st.sidebar.checkbox("Set filter", False)
if has_filter:
    filter_property = st.sidebar.selectbox("Filter property", _PROPERTIES, 0)
    filter_max = df_results[filter_property].max().item()
    filter_value = st.sidebar.slider(
        "Filter cutoff (less than)", 0.0, filter_max, filter_max
    )
    df_results = df_results[(df_results[filter_property] <= filter_value)]

filter_recently = st.sidebar.slider(
    "Recent results (minutes)",
    value=60,
    min_value=1,
    max_value=60,
    step=1,
    format="%dm",
)
dt_filter_recently = dt_now - pd.Timedelta(filter_recently, unit="minutes")

workflow_selected = st.selectbox("Select workflow", workflow_names)
models_selected = st.multiselect("Select models", model_names, model_names)

df_display = df_results[
    (df_results.workflow.isin([workflow_selected]))
    & (df_results.model.isin(models_selected))
    & (df_results.started_at > dt_filter_recently)
].sort_values("started_at", ascending=False)

df_display_table = df_display[_BASE_COLUMNS + properties_selected]

if has_plot:
    st.vega_lite_chart(
        df_display,
        {
            "height": 400,
            "mark": {"type": "point", "opacity": 1.0, "filled": True, "size": 50},
            "encoding": {
                "x": {
                    "field": "started_at",
                    "type": "temporal",
                    "title": "Started at",
                    "axis": {
                        "format": "%H:%M",
                        "labelAngle": -45,
                        "tickSize": 10,
                        "bandPosition": 0,
                        "gridDash": {
                            "condition": {
                                "test": {
                                    "field": "value",
                                    "timeUnit": "minute",
                                    "equal": 0,
                                },
                                "value": [],
                            },
                            "value": [2, 2],
                        },
                        "tickDash": {
                            "condition": {
                                "test": {
                                    "field": "value",
                                    "timeUnit": "minute",
                                    "equal": 0,
                                },
                                "value": [],
                            },
                            "value": [2, 2],
                        },
                    },
                },
                "y": {
                    "field": plot_property_selected,
                    "type": "quantitative",
                    "title": plot_property_selected.replace("_", " "),
                },
                "color": {"field": "model", "type": "nominal"},
                "shape": {"field": "model", "type": "nominal"},
                "tooltip": {"field": plot_property_selected, "type": "quantitive"},
            },
        },
        True,
    )

st.table(df_display_table.reset_index(drop=True))
