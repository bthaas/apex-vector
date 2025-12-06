# ApexVector

ApexVector is a high-performance, persistent, approximate nearest neighbor (ANN) search vector database written in Rust. It implements the HNSW (Hierarchical Navigable Small World) algorithm with scalar quantization for memory efficiency.

![Build Status](https://img.shields.io/badge/build-passing-brightgreen)
![License](https://img.shields.io/badge/license-MIT-blue)

## Features

- **HNSW Indexing**: Fast approximate nearest neighbor search using multi-layer graphs.
- **Scalar Quantization**: Compresses 32-bit floating point vectors to 8-bit integers (4x RAM reduction).
- **Persistence**: ACID-compliant durability using Write-Ahead Logging (WAL) and snapshots.
- **gRPC Interface**: High-performance API using Tonic/Protobuf.
- **Distance Metrics**: Euclidean (L2) and Cosine Similarity support.

## Project Structure

- `src/distance.rs`: SIMD-friendly distance implementations.
- `src/hnsw.rs`: Core graph algorithm (Insert, Search, Layering).
- `src/quantizer.rs`: Logic for compressing/decompressing vectors.
- `src/storage.rs`: Disk I/O management (WAL, Snapshots).
- `src/store.rs`: Naive baseline for correctness verification.
- `src/main.rs`: gRPC server entry point.
- `src/bin/benchmark.rs`: Performance testing suite.

## Getting Started

### Prerequisites

- Rust (latest stable)
- `protoc` (Protocol Buffers compiler)

### Building

```bash
cargo build --release
```

### Running the Server

Start the gRPC server (listens on `[::1]:50051`):

```bash
cargo run --release --bin apexvector
```

### API Usage

You can use `grpcurl` to interact with the server:

**Upsert a Vector:**
```bash
grpcurl -plaintext -d '{"id": 1, "vector": [0.1, 0.5, 0.9, ...]}' [::1]:50051 apexvector.VectorService/Upsert
```

**Query Nearest Neighbors:**
```bash
grpcurl -plaintext -d '{"k": 5, "vector": [0.1, 0.5, 0.9, ...]}' [::1]:50051 apexvector.VectorService/Query
```

## Benchmarks

Run the included benchmark suite to compare HNSW against a linear search baseline. It generates synthetic data if SIFT1M is not found.

```bash
cargo run --release --bin benchmark
```

Expected Results (Synthetic 10k):
- Recall@10: >95%
- Latency Improvement: >10x vs Linear Search
- Build Time: <1s

## Configuration

The default configuration (in `src/main.rs`) uses:
- **Dimensions**: 128 (automatically inferred from first insert or defaults)
- **M**: 16 (Max neighbors per node)
- **ef_construction**: 200 (Search depth during build)

## License

MIT
