#!/usr/bin/env bash

set -o errexit
set -o nounset
set -o pipefail
set -o errtrace

function usage() {
    cat << EOF
Usage:
    generate_protobuf.sh <directory>

Example:
    ./generate_secrets.sh ./app/src/components/postgres_example_gen/proto

EOF
}

if ! command -v protoc &> /dev/null; then
    echo "Error: protoc not found" >&2
    exit 1
fi

PROTOBUF_SRC_DIR="${1-}"

if [ -z "${PROTOBUF_SRC_DIR}" ]; then
    echo "Error: directory parameter missing"
    usage
    exit 1
fi

for f in "${PROTOBUF_SRC_DIR}"/*.proto; do
    echo "Generating Python wrapper from protobuf defintion: ${f}"
    protoc -I. --python_out=. "$f"
done
