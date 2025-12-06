use crate::distance::{DistanceMetric, Euclidean};
use crate::quantizer::Quantizer;
use rand::Rng;
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashSet};
use serde::{Serialize, Deserialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Node {
    pub vector_id: usize,
    pub vector: Vec<u8>, // Compressed
    pub layers: Vec<Vec<usize>>,
    pub zero_layer_neighbors: Vec<usize>,
}

impl Node {
    fn new(vector_id: usize, vector: Vec<u8>, max_layer: usize) -> Self {
        Node {
            vector_id,
            vector,
            layers: vec![Vec::new(); max_layer + 1],
            zero_layer_neighbors: Vec::new(),
        }
    }
}

#[derive(Serialize, Deserialize)]
pub struct HnswIndex {
    pub nodes: Vec<Node>,
    pub quantizer: Quantizer,
    pub entry_point: Option<usize>,
    pub max_layer: usize,
    pub m: usize,
    pub ef_construction: usize,
    pub level_mult: f64,
}

#[derive(PartialEq)]
struct Candidate {
    distance: f32,
    node_index: usize,
}

impl Eq for Candidate {}

impl Ord for Candidate {
    fn cmp(&self, other: &Self) -> Ordering {
        self.distance.partial_cmp(&other.distance).unwrap_or(Ordering::Equal)
    }
}

impl PartialOrd for Candidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

#[derive(PartialEq)]
struct ReverseCandidate(Candidate);

impl Eq for ReverseCandidate {}

impl Ord for ReverseCandidate {
    fn cmp(&self, other: &Self) -> Ordering {
        other.0.cmp(&self.0)
    }
}

impl PartialOrd for ReverseCandidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl HnswIndex {
    pub fn new(m: usize, ef_construction: usize, quantizer: Quantizer) -> Self {
        HnswIndex {
            nodes: Vec::new(),
            quantizer,
            entry_point: None,
            max_layer: 0,
            m,
            ef_construction,
            level_mult: 1.0 / (m as f64).ln(),
        }
    }

    fn get_random_layer(&self) -> usize {
        let mut rng = rand::thread_rng();
        let r: f64 = rng.gen();
        ((-r.ln()) * self.level_mult).floor() as usize
    }

    // Distance now calculates between Query (f32) and Node (u8)
    fn dist(&self, query: &[f32], node_vec_u8: &[u8]) -> f32 {
        // Decompress on the fly
        // Optimization: In real SIMD code, we might do L2 on u8 directly (if we had u8 query) 
        // or fast unpack. Here we use the quantizer's decompress.
        let node_vec_f32 = self.quantizer.decompress(node_vec_u8);
        Euclidean.calculate(query, &node_vec_f32)
    }

    fn search_layer(
        &self, 
        query: &[f32], 
        entry_point: usize, 
        layer: usize, 
        ef: usize
    ) -> BinaryHeap<Candidate> {
        let mut visited = HashSet::new();
        let mut candidates = BinaryHeap::new(); 
        let mut results = BinaryHeap::new();

        let dist = self.dist(query, &self.nodes[entry_point].vector);
        
        visited.insert(entry_point);
        candidates.push(ReverseCandidate(Candidate { distance: dist, node_index: entry_point }));
        results.push(Candidate { distance: dist, node_index: entry_point });

        while let Some(ReverseCandidate(calc)) = candidates.pop() {
            let current_dist = calc.distance;
            let current_node_idx = calc.node_index;

            if let Some(furthest) = results.peek() {
                if current_dist > furthest.distance && results.len() >= ef {
                    break;
                }
            }

            let neighbors = if layer == 0 && !self.nodes[current_node_idx].zero_layer_neighbors.is_empty() {
                &self.nodes[current_node_idx].zero_layer_neighbors
            } else {
                &self.nodes[current_node_idx].layers[layer]
            };

            for &neighbor_idx in neighbors {
                if !visited.contains(&neighbor_idx) {
                    visited.insert(neighbor_idx);
                    let neighbor_dist = self.dist(query, &self.nodes[neighbor_idx].vector);
                    
                    if results.len() < ef || neighbor_dist < results.peek().unwrap().distance {
                        candidates.push(ReverseCandidate(Candidate { distance: neighbor_dist, node_index: neighbor_idx }));
                        results.push(Candidate { distance: neighbor_dist, node_index: neighbor_idx });
                        
                        if results.len() > ef {
                            results.pop();
                        }
                    }
                }
            }
        }

        results
    }

    pub fn search(&self, query: &[f32], k: usize) -> Vec<(usize, f32)> {
        if self.nodes.is_empty() {
            return Vec::new();
        }

        let mut curr_entry = self.entry_point.unwrap();
        let mut curr_dist = self.dist(query, &self.nodes[curr_entry].vector);

        for level in (1..=self.max_layer).rev() {
            let mut changed = true;
            while changed {
                changed = false;
                let neighbors = &self.nodes[curr_entry].layers[level];
                for &neighbor_idx in neighbors {
                    let d = self.dist(query, &self.nodes[neighbor_idx].vector);
                    if d < curr_dist {
                        curr_dist = d;
                        curr_entry = neighbor_idx;
                        changed = true;
                    }
                }
            }
        }

        let ef = k.max(self.ef_construction);
        let top_candidates = self.search_layer(query, curr_entry, 0, ef);

        let mut final_results: Vec<(usize, f32)> = top_candidates.into_iter()
            .map(|c| (self.nodes[c.node_index].vector_id, c.distance))
            .collect();
        
        final_results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
        final_results.into_iter().take(k).collect()
    }

    pub fn insert(&mut self, vector: Vec<f32>, id: usize) {
         // Compress BEFORE insertion logic
         let compressed_vec = self.quantizer.compress(&vector);
         
         let target_level = self.get_random_layer();
         let new_id = self.nodes.len();
         
         // Store compressed vector
         let mut new_node = Node::new(id, compressed_vec, target_level);
         
         if self.nodes.is_empty() {
             self.nodes.push(new_node);
             self.entry_point = Some(0);
             self.max_layer = target_level;
             return;
         }
         
         let mut curr_entry = self.entry_point.unwrap();
         let mut curr_dist = self.dist(&vector, &self.nodes[curr_entry].vector);
         
         for l in (target_level + 1..=self.max_layer).rev() {
             let mut changed = true;
             while changed {
                 changed = false;
                 if l >= self.nodes[curr_entry].layers.len() { continue; }
                 
                 for &neighbor_idx in &self.nodes[curr_entry].layers[l] {
                     let d = self.dist(&vector, &self.nodes[neighbor_idx].vector);
                     if d < curr_dist {
                         curr_dist = d;
                         curr_entry = neighbor_idx;
                         changed = true;
                     }
                 }
             }
         }
         
         let mut neighbors_by_level: Vec<Vec<usize>> = vec![Vec::new(); target_level + 1];
         let mut search_entry = curr_entry;
         
         for l in (0..=target_level).rev() {
             let candidates = self.search_layer_internal(&vector, search_entry, l, self.ef_construction);
             
             let mut sorted: Vec<(f32, usize)> = candidates.into_iter()
                .map(|c| (c.distance, c.node_index))
                .collect();
             sorted.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
             
             let selected: Vec<usize> = sorted.iter().take(self.m).map(|x| x.1).collect();
             
             if let Some(&first) = selected.first() {
                 search_entry = first;
             }
             
             neighbors_by_level[l] = selected;
         }
         
         for (l, neighbors) in neighbors_by_level.iter().enumerate() {
             new_node.layers[l] = neighbors.clone();
             if l == 0 {
                 new_node.zero_layer_neighbors = neighbors.clone();
             }
         }
         
         self.nodes.push(new_node);
         
         for (l, neighbors) in neighbors_by_level.into_iter().enumerate() {
             for neighbor_idx in neighbors {
                 self.nodes[neighbor_idx].layers[l].push(new_id);
                 if l == 0 {
                     self.nodes[neighbor_idx].zero_layer_neighbors.push(new_id);
                 }
             }
         }
         
         if target_level > self.max_layer {
             self.max_layer = target_level;
             self.entry_point = Some(new_id);
         }
    }

    fn search_layer_internal(&self, query: &[f32], entry: usize, layer: usize, ef: usize) -> BinaryHeap<Candidate> {
        self.search_layer(query, entry, layer, ef)
    }
}
