FROM pgvector/pgvector:pg16

ARG AGE_REF=PG16

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        bison \
        ca-certificates \
        flex \
        git \
        postgresql-server-dev-16 \
    && git clone --depth 1 --branch "${AGE_REF}" https://github.com/apache/age.git /tmp/age \
    && make -C /tmp/age PG_CONFIG=/usr/lib/postgresql/16/bin/pg_config install \
    && rm -rf /tmp/age /var/lib/apt/lists/*

COPY initdb/ /docker-entrypoint-initdb.d/
