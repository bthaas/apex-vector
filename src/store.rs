use crate::distance::DistanceMetric;
use std::cmp::Ordering;

pub struct VectorStore {
    // Storing as (id, vector)
    pub data: Vec<(usize, Vec<f32>)>,
}

impl VectorStore {
    pub fn new() -> Self {
        VectorStore { data: Vec::new() }
    }

    pub fn add(&mut self, id: usize, vector: Vec<f32>) {
        self.data.push((id, vector));
    }

    pub fn search(&self, query: &[f32], k: usize, metric: &dyn DistanceMetric) -> Vec<(usize, f32)> {
        let mut distances: Vec<(usize, f32)> = self.data
            .iter()
            .map(|(id, vector)| {
                let dist = metric.calculate(query, vector);
                (*id, dist)
            })
            .collect();

        // Sort by distance (ascending)
        distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));

        // Return top k
        distances.into_iter().take(k).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distance::Euclidean;
    use std::time::Instant;
    use rand::Rng; // We might not have rand in Cargo.toml yet, let's check or use simple pseudo-random

    // Simple pseudo-random generator to avoid adding dependencies if possible, 
    // or we can add rand to Cargo.toml.
    // The prompt implied we might have access to standard crates. 
    // I'll update Cargo.toml to include `rand` for generation.

    #[test]
    fn test_linear_search_correctness() {
        let mut store = VectorStore::new();
        store.add(1, vec![1.0, 0.0]);
        store.add(2, vec![10.0, 0.0]);
        store.add(3, vec![0.0, 1.0]);

        let query = vec![0.0, 0.0];
        let metric = Euclidean;
        
        let results = store.search(&query, 2, &metric);
        
        // Expected: 1 (dist 1.0) and 3 (dist 1.0) are closest. 2 is dist 10.0.
        assert_eq!(results.len(), 2);
        assert!(results.iter().any(|(id, _)| *id == 1));
        assert!(results.iter().any(|(id, _)| *id == 3));
    }

    #[test]
    fn benchmark_10k_vectors() {
        let mut store = VectorStore::new();
        let dim = 128;
        let num_vectors = 10_000;
        
        // Deterministic generation for consistency
        for i in 0..num_vectors {
            // Simple pattern: vector [i as f32, 0, ..., 0] normalized or just randomish?
            // Let's make them random-ish using a simple LCG or just values
            let val = (i as f32) / 1000.0;
            let vec = vec![val; dim]; 
            store.add(i, vec);
        }

        let query = vec![0.5; dim];
        let metric = Euclidean;
        
        let start = Instant::now();
        let _results = store.search(&query, 10, &metric);
        let duration = start.elapsed();

        println!("Linear search for 10,000 vectors (dim={}) took: {:?}", dim, duration);
        
        // Sanity check
        assert!(duration.as_secs() < 5, "Should be much faster than 5s");
    }
}
