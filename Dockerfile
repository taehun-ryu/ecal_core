FROM ubuntu:22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       build-essential \
       cmake \
       pkg-config \
       libopencv-dev \
       libceres-dev \
       libhdf5-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace/ecal
COPY . .

RUN cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DECAL_STANDALONE=ON \
    && cmake --build build -j"$(nproc)"

FROM ubuntu:22.04 AS runtime

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       libopencv-dev \
       libceres-dev \
       libhdf5-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY --from=builder /workspace/ecal/build/calibration_main /usr/local/bin/calibration_main
COPY --from=builder /workspace/ecal/config /app/config

ENTRYPOINT ["/usr/local/bin/calibration_main"]
CMD ["/app/config/calibration.docker.yaml"]
