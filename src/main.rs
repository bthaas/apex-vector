use tonic::{transport::Server, Request, Response, Status};
// Must match the proto package name
use apexvector::vector_service_server::{VectorService, VectorServiceServer};
use apexvector::{UpsertRequest, UpsertResponse, QueryRequest, QueryResponse, SearchResult};

use apex_vector::hnsw::HnswIndex;
use apex_vector::quantizer::Quantizer;
use apex_vector::storage::{StorageManager, WalEntry};
use std::sync::{Arc, RwLock, Mutex};
use std::path::Path;

pub mod apexvector {
    tonic::include_proto!("apexvector");
}

pub struct MyVectorService {
    index: Arc<RwLock<HnswIndex>>,
    storage: Arc<Mutex<StorageManager>>,
}

#[tonic::async_trait]
impl VectorService for MyVectorService {
    async fn upsert(
        &self,
        request: Request<UpsertRequest>,
    ) -> Result<Response<UpsertResponse>, Status> {
        let req = request.into_inner();
        let vector = req.vector;
        let id = req.id as usize;

        // 1. Wal Append
        {
            let mut storage_guard = self.storage.lock().unwrap();
            let entry = WalEntry::Insert { id, vector: vector.clone() };
            if let Err(e) = storage_guard.append_wal(entry) {
                return Err(Status::internal(format!("WAL write failed: {}", e)));
            }
        }

        // 2. Memory Insert
        {
            let mut index_guard = self.index.write().unwrap();
            // Assuming index is initialized with a quantizer.
            // If dim mismatch, might panic, so check dim?
            // HNSW handles raw vector -> compress -> insert.
            index_guard.insert(vector, id);
        }

        Ok(Response::new(UpsertResponse { success: true }))
    }

    async fn query(
        &self,
        request: Request<QueryRequest>,
    ) -> Result<Response<QueryResponse>, Status> {
        let req = request.into_inner();
        let vector = req.vector;
        let k = req.k as usize;

        let results = {
            let index_guard = self.index.read().unwrap();
            index_guard.search(&vector, k)
        };

        let response_results: Vec<SearchResult> = results.into_iter()
            .map(|(id, dist)| SearchResult {
                id: id as u32,
                distance: dist,
            })
            .collect();

        Ok(Response::new(QueryResponse { results: response_results }))
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let addr = "[::1]:50051".parse()?;
    let data_path = Path::new("./data");
    
    // 1. Initialize Storage
    let mut storage = StorageManager::new(data_path)?;
    
    // 2. Recover Index
    // specific logic: Load Snapshot -> Replay WAL
    let mut index = if let Some(loaded_index) = storage.load_snapshot()? {
        println!("Loaded snapshot.");
        loaded_index
    } else {
        println!("Creating new index.");
        // We need a quantizer. For functionality without distinct training phase, 
        // we'll hack a default quantizer for 128-dim vectors (standard SIFT).
        // WARNING: Fixed dimension here for new index.
        let dim = 128;
        let mut q = Quantizer::new();
        // Initialize huge bounds to allow insertion without strict training
        // This ruins quantization precision but allows the server to run.
        q.min = vec![-1000.0; dim];
        q.max = vec![1000.0; dim];
        
        HnswIndex::new(16, 200, q)
    };
    
    // Replay WAL
    let wal_entries = storage.read_wal()?;
    if !wal_entries.is_empty() {
        println!("Replaying {} WAL entries...", wal_entries.len());
        for entry in wal_entries {
            match entry {
                WalEntry::Insert { id, vector } => {
                    index.insert(vector, id);
                }
            }
        }
    }

    let service = MyVectorService {
        index: Arc::new(RwLock::new(index)),
        storage: Arc::new(Mutex::new(storage)),
    };

    println!("ApexVector server listening on {}", addr);

    Server::builder()
        .add_service(VectorServiceServer::new(service))
        .serve(addr)
        .await?;

    Ok(())
}
