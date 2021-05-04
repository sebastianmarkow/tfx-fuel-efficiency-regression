#!/usr/bin/env bash

set -o errexit
set -o nounset
set -o pipefail
set -o errtrace

csvsql -S -i postgresql "${CSVFILE_TO_LOAD}"
csvsql -S --insert --db "postgresql+psycopg2://${POSTGRES_USER}:${POSTGRES_PASSWORD}@${POSTGRES_HOST}:${POSTGRES_PORT}/${POSTGRES_DATABASE}" --overwrite  "${CSVFILE_TO_LOAD}"
