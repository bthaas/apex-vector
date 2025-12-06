pub trait DistanceMetric {
    fn calculate(&self, a: &[f32], b: &[f32]) -> f32;
}

pub struct Euclidean;

impl DistanceMetric for Euclidean {
    fn calculate(&self, a: &[f32], b: &[f32]) -> f32 {
        if a.len() != b.len() {
            panic!("Vectors must have the same length");
        }
        
        // Rust's auto-vectorization is often good enough for standard loops,
        // especially with iterators. 
        a.iter()
         .zip(b.iter())
         .map(|(x, y)| (x - y).powi(2))
         .sum::<f32>()
         .sqrt()
    }
}

pub struct Cosine;

impl DistanceMetric for Cosine {
    fn calculate(&self, a: &[f32], b: &[f32]) -> f32 {
        if a.len() != b.len() {
            panic!("Vectors must have the same length");
        }

        let dot_product: f32 = a.iter()
            .zip(b.iter())
            .map(|(x, y)| x * y)
            .sum();

        let magnitude_a: f32 = a.iter().map(|x| x.powi(2)).sum::<f32>().sqrt();
        let magnitude_b: f32 = b.iter().map(|x| x.powi(2)).sum::<f32>().sqrt();

        if magnitude_a == 0.0 || magnitude_b == 0.0 {
            return 0.0;
        }

        dot_product / (magnitude_a * magnitude_b)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array1;
    use ndarray::prelude::*;

    #[test]
    fn test_euclidean_distance() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];
        let metric = Euclidean;
        let result = metric.calculate(&a, &b);

        // Expected: sqrt((1-4)^2 + (2-5)^2 + (3-6)^2) = sqrt(9+9+9) = sqrt(27) ≈ 5.196
        let expected = (27.0 as f32).sqrt();
        assert!((result - expected).abs() < 1e-5);
    }

    #[test]
    fn test_cosine_similarity() {
        let a = vec![1.0, 0.0, -1.0];
        let b = vec![0.0, 1.0, 0.0]; // Orthogonal
        let metric = Cosine;
        let result = metric.calculate(&a, &b);
        assert!((result - 0.0).abs() < 1e-5);

        let a = vec![1.0, 1.0];
        let b = vec![1.0, 1.0]; // Identical direction
        let result = metric.calculate(&a, &b);
        assert!((result - 1.0).abs() < 1e-5);
    }
    
    #[test]
    fn test_compare_with_ndarray() {
        let a_data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let b_data = vec![5.0, 4.0, 3.0, 2.0, 1.0];
        
        let a_arr = Array1::from(a_data.clone());
        let b_arr = Array1::from(b_data.clone());
        
        // Euclidean
        let my_euclidean = Euclidean.calculate(&a_data, &b_data);
        // ndarray L2 norm of diff
        let diff = &a_arr - &b_arr;
        let nd_euclidean = diff.mapv(|x| x.powi(2)).sum().sqrt();
        
        assert!((my_euclidean - nd_euclidean).abs() < 1e-5, "Euclidean mismatch: {} vs {}", my_euclidean, nd_euclidean);

        // Cosine 
        // Cosine Similarity = Dot(A, B) / (|A| * |B|)
        let my_cosine = Cosine.calculate(&a_data, &b_data);
        
        let dot = a_arr.dot(&b_arr);
        let norm_a = a_arr.mapv(|x| x.powi(2)).sum().sqrt();
        let norm_b = b_arr.mapv(|x| x.powi(2)).sum().sqrt();
        let nd_cosine = dot / (norm_a * norm_b);
        
        assert!((my_cosine - nd_cosine).abs() < 1e-5, "Cosine mismatch: {} vs {}", my_cosine, nd_cosine);
    }
}
