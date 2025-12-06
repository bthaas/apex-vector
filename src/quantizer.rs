use serde::{Serialize, Deserialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Quantizer {
    pub min: Vec<f32>,
    pub max: Vec<f32>,
}

impl Quantizer {
    pub fn new() -> Self {
        Quantizer {
            min: Vec::new(),
            max: Vec::new(),
        }
    }

    pub fn train(&mut self, vectors: &[Vec<f32>]) {
        if vectors.is_empty() {
            return;
        }

        let dim = vectors[0].len();
        self.min = vec![f32::MAX; dim];
        self.max = vec![f32::MIN; dim];

        for vec in vectors {
            for (i, &val) in vec.iter().enumerate() {
                if val < self.min[i] {
                    self.min[i] = val;
                }
                if val > self.max[i] {
                    self.max[i] = val;
                }
            }
        }
    }

    pub fn compress(&self, vector: &[f32]) -> Vec<u8> {
        if self.min.is_empty() {
            // Fallback or panic? Assuming trained.
            // If not trained, maybe return 0s or panic. 
            // Let's assume user calls train first.
            panic!("Quantizer not trained");
        }
        
        vector.iter().enumerate().map(|(i, &val)| {
            let range = self.max[i] - self.min[i];
            if range == 0.0 {
                return 0;
            }
            let normalized = (val - self.min[i]) / range;
            let clamped = normalized.clamp(0.0, 1.0);
            (clamped * 255.0) as u8
        }).collect()
    }

    pub fn decompress(&self, vector: &[u8]) -> Vec<f32> {
        vector.iter().enumerate().map(|(i, &val)| {
            let range = self.max[i] - self.min[i];
            self.min[i] + (val as f32 / 255.0) * range
        }).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quantization_cycle() {
        let data = vec![
            vec![1.0, 10.0, -5.0],
            vec![0.0, 5.0, 5.0],
            vec![2.0, 0.0, 0.0],
        ];

        let mut q = Quantizer::new();
        q.train(&data);

        // Min: [0.0, 0.0, -5.0]
        // Max: [2.0, 10.0, 5.0]
        
        assert_eq!(q.min, vec![0.0, 0.0, -5.0]);
        assert_eq!(q.max, vec![2.0, 10.0, 5.0]);

        let vec = vec![1.0, 5.0, 0.0];
        // 1.0 is mid of 0..2 -> 127
        // 5.0 is mid of 0..10 -> 127
        // 0.0 is mid of -5..5 -> 127
        
        let compressed = q.compress(&vec);
        let decompressed = q.decompress(&compressed);

        // Allow some error due to int conversion
        for (v, d) in vec.iter().zip(decompressed.iter()) {
            assert!((v - d).abs() < 0.1, "Original: {}, Received: {}", v, d);
        }
    }
}
