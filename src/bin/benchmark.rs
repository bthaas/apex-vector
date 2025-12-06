use apex_vector::distance::{DistanceMetric, Euclidean};
use apex_vector::hnsw::HnswIndex;
use apex_vector::quantizer::Quantizer;
use apex_vector::store::VectorStore;
use std::fs::File;
use std::io::{self, Read, BufReader};
use std::time::Instant;
use std::path::Path;

// Helper to read fvecs (SIFT format)
// Format per vector: <dim (int32)> <f32> <f32> ...
fn read_fvecs(path: &str, max_vecs: usize) -> io::Result<Vec<Vec<f32>>> {
    let file = File::open(path)?;
    let mut reader = BufReader::new(file);
    let mut vectors = Vec::new();
    
    let mut dim_buf = [0u8; 4];
    while vectors.len() < max_vecs {
        if let Err(_) = reader.read_exact(&mut dim_buf) {
            break; // EOF
        }
        let dim = i32::from_le_bytes(dim_buf) as usize;
        
        let mut vec_buf = vec![0u8; dim * 4];
        reader.read_exact(&mut vec_buf)?;
        
        let mut vec = Vec::with_capacity(dim);
        for chunk in vec_buf.chunks(4) {
            let val = f32::from_le_bytes(chunk.try_into().unwrap());
            vec.push(val);
        }
        vectors.push(vec);
    }
    Ok(vectors)
}

fn generate_synthetic(n: usize, dim: usize) -> Vec<Vec<f32>> {
    println!("Generating {} synthetic vectors (dim={})...", n, dim);
    let mut vectors = Vec::with_capacity(n);
    // Simple PRNG
    let mut val = 0.5;
    for _ in 0..n {
        let mut vec = Vec::with_capacity(dim);
        for _ in 0..dim {
            val = (val * 3.14159 + 1.234) % 10.0;
            vec.push(val);
        }
        vectors.push(vec);
    }
    vectors
}

fn compute_recall(ground_truth: &[(usize, f32)], results: &[(usize, f32)]) -> f32 {
    let k = ground_truth.len();
    if k == 0 { return 0.0; }
    
    let gt_ids: std::collections::HashSet<usize> = ground_truth.iter().map(|(id, _)| *id).collect();
    let hits = results.iter().filter(|(id, _)| gt_ids.contains(id)).count();
    
    hits as f32 / k as f32
}

fn main() {
    println!("=== ApexVector Benchmark Suite ===");
    
    // Config
    let dim = 128;
    let n_base = 10_000; // Use small subset for quick demo, set to 1_000_000 for full logic
    let n_query = 100;
    
    // 1. Data Loading
    let base_vectors = if Path::new("sift/sift_base.fvecs").exists() {
        println!("Loading SIFT1M base...");
        read_fvecs("sift/sift_base.fvecs", 1_000_000).unwrap_or_else(|_| generate_synthetic(n_base, dim))
    } else {
        generate_synthetic(n_base, dim)
    };
    
    let query_vectors = generate_synthetic(n_query, base_vectors[0].len()); 
    // Ideally use sift_query.fvecs if available, but for fallback synthetic we match dim.

    let k = 10;
    
    println!("Dataset: {} vectors, dim {}", base_vectors.len(), base_vectors[0].len());
    
    // 2. Linear Search (Ground Truth)
    println!("\n--- Phase 2: Naive Baseline (Linear) ---");
    let mut store = VectorStore::new();
    for (i, vec) in base_vectors.iter().enumerate() {
        store.add(i, vec.clone());
    }
    
    let start_linear = Instant::now();
    let mut linear_results = Vec::new();
    for query in &query_vectors {
        linear_results.push(store.search(query, k, &Euclidean));
    }
    let duration_linear = start_linear.elapsed();
    println!("Linear Search Latency (avg): {:.4}ms", (duration_linear.as_micros() as f32 / n_query as f32) / 1000.0);
    
    // 3. HNSW Build
    println!("\n--- Phase 3+4: HNSW Build (Quantized) ---");
    // Train Quantizer
    let mut quantizer = Quantizer::new();
    quantizer.train(&base_vectors[0..std::cmp::min(1000, base_vectors.len())]); // Train on subset
    
    let build_start = Instant::now();
    let mut hnsw = HnswIndex::new(16, 100, quantizer); // M=16, ef_con=100
    for (i, vec) in base_vectors.iter().enumerate() {
        if i % 1000 == 0 {
            // printStatus?
        }
        hnsw.insert(vec.clone(), i);
    }
    println!("HNSW Build Time: {:.2?}", build_start.elapsed());
    
    // 4. HNSW Query
    println!("\n--- Phase 7: HNSW Query & Verify ---");
    let start_hnsw = Instant::now();
    let mut hnsw_results = Vec::new();
    for query in &query_vectors {
        hnsw_results.push(hnsw.search(query, k));
    }
    let duration_hnsw = start_hnsw.elapsed();
    let avg_latency = (duration_hnsw.as_micros() as f32 / n_query as f32) / 1000.0;
    println!("HNSW Search Latency (avg): {:.4}ms", avg_latency);
    
    // 5. Comparison
    let mut total_recall = 0.0;
    for (gt, res) in linear_results.iter().zip(hnsw_results.iter()) {
        total_recall += compute_recall(gt, res);
    }
    let avg_recall = total_recall / n_query as f32;
    
    println!("\n=== Results ===");
    println!("Recall@{}: {:.2}%", k, avg_recall * 100.0);
    println!("Latency Improvement: {:.1}x", (duration_linear.as_micros() as f32 / duration_hnsw.as_micros() as f32));
    
    // RAM estimation (Simulated)
    // Raw: N * Dim * 4 bytes
    // HNSW: N * Dim * 1 byte (quantized) + N * M * 4 bytes (neighbors) * levels (avg)
    // + overhead
    println!("Memory (Theoretical) reduced by ~4x due to Scalar Quantization.");
}
