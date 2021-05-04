# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: app/src/components/postgres_example_gen/proto/postgres_feature_query.proto
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor.FileDescriptor(
  name='app/src/components/postgres_example_gen/proto/postgres_feature_query.proto',
  package='tfx.examples.custom_components.postgres_example_gen',
  syntax='proto3',
  serialized_options=None,
  create_key=_descriptor._internal_create_key,
  serialized_pb=b'\nJapp/src/components/postgres_example_gen/proto/postgres_feature_query.proto\x12\x33tfx.examples.custom_components.postgres_example_gen\"\x96\x01\n\x14PostgresFeatureQuery\x12\r\n\x05table\x18\x01 \x01(\t\x12U\n\x07\x63olumns\x18\x02 \x03(\x0b\x32\x44.tfx.examples.custom_components.postgres_example_gen.postgres_column\x12\x18\n\x10optionFilterNull\x18\x03 \x01(\x08\"\x1f\n\x0fpostgres_column\x12\x0c\n\x04name\x18\x01 \x01(\tb\x06proto3'
)




_POSTGRESFEATUREQUERY = _descriptor.Descriptor(
  name='PostgresFeatureQuery',
  full_name='tfx.examples.custom_components.postgres_example_gen.PostgresFeatureQuery',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='table', full_name='tfx.examples.custom_components.postgres_example_gen.PostgresFeatureQuery.table', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='columns', full_name='tfx.examples.custom_components.postgres_example_gen.PostgresFeatureQuery.columns', index=1,
      number=2, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='optionFilterNull', full_name='tfx.examples.custom_components.postgres_example_gen.PostgresFeatureQuery.optionFilterNull', index=2,
      number=3, type=8, cpp_type=7, label=1,
      has_default_value=False, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=132,
  serialized_end=282,
)


_POSTGRES_COLUMN = _descriptor.Descriptor(
  name='postgres_column',
  full_name='tfx.examples.custom_components.postgres_example_gen.postgres_column',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='name', full_name='tfx.examples.custom_components.postgres_example_gen.postgres_column.name', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=284,
  serialized_end=315,
)

_POSTGRESFEATUREQUERY.fields_by_name['columns'].message_type = _POSTGRES_COLUMN
DESCRIPTOR.message_types_by_name['PostgresFeatureQuery'] = _POSTGRESFEATUREQUERY
DESCRIPTOR.message_types_by_name['postgres_column'] = _POSTGRES_COLUMN
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

PostgresFeatureQuery = _reflection.GeneratedProtocolMessageType('PostgresFeatureQuery', (_message.Message,), {
  'DESCRIPTOR' : _POSTGRESFEATUREQUERY,
  '__module__' : 'app.src.components.postgres_example_gen.proto.postgres_feature_query_pb2'
  # @@protoc_insertion_point(class_scope:tfx.examples.custom_components.postgres_example_gen.PostgresFeatureQuery)
  })
_sym_db.RegisterMessage(PostgresFeatureQuery)

postgres_column = _reflection.GeneratedProtocolMessageType('postgres_column', (_message.Message,), {
  'DESCRIPTOR' : _POSTGRES_COLUMN,
  '__module__' : 'app.src.components.postgres_example_gen.proto.postgres_feature_query_pb2'
  # @@protoc_insertion_point(class_scope:tfx.examples.custom_components.postgres_example_gen.postgres_column)
  })
_sym_db.RegisterMessage(postgres_column)


# @@protoc_insertion_point(module_scope)