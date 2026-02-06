FROM ghcr.io/berriai/litellm:main

# RUN pip install --no-cache-dir --upgrade \
#     "litellm==1.77.0" \
#     "litellm[proxy]==1.77.0" \
#     "opentelemetry-api==1.38.0" \
#     "opentelemetry-sdk==1.38.0" \
#     "opentelemetry-exporter-otlp-proto-grpc==1.38.0" \
#     "opentelemetry-exporter-otlp==1.38.0" \
#     "opentelemetry-distro==0.59b0" \
#     "opentelemetry-instrumentation==0.59b0"

# Generate Prisma binaries for database support
RUN prisma generate --schema=/usr/local/lib/python3.9/site-packages/litellm/proxy/schema.prisma
