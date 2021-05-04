ARG PYTHON_VERSION=3.7.10
ARG TINI_VERSION=0.19.0

FROM docker.io/python:${PYTHON_VERSION}-slim-buster

ARG TINI_VERSION

RUN apt-get update \
 && apt-get install --upgrade --yes --no-install-recommends \
    ca-certificates \
    curl \
    tzdata \
 && apt-get autoremove --yes --purge \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/* \
 && update-ca-certificates \
 && ln -sf /usr/share/zoneinfo/UTC /etc/localtime

RUN curl -L -o /usr/bin/tini "https://github.com/krallin/tini/releases/download/v${TINI_VERSION}/tini-static-amd64" \
 && chown root:root /usr/bin/tini \
 && chmod 0755 /usr/bin/tini \
 && /usr/bin/tini --version

COPY ./requirements.txt /opt/requirements.txt

RUN pip install --no-cache-dir --requirement /opt/requirements.txt

WORKDIR /opt/app
COPY ./src /opt/app

ENTRYPOINT ["/usr/bin/tini", "--"]
