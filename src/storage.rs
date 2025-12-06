use crate::hnsw::HnswIndex;
use serde::{Serialize, Deserialize};
use std::fs::{File, OpenOptions};
use std::io::{self, BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};

#[derive(Serialize, Deserialize, Debug)]
pub enum WalEntry {
    Insert { id: usize, vector: Vec<f32> },
}

pub struct StorageManager {
    wal_path: PathBuf,
    snapshot_path: PathBuf,
    wal_file: Option<BufWriter<File>>,
}

impl StorageManager {
    pub fn new(base_path: &Path) -> io::Result<Self> {
        let wal_path = base_path.join("wal.bin");
        let snapshot_path = base_path.join("snapshot.bin");
        
        // Ensure directory exists
        if let Some(parent) = base_path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::create_dir_all(base_path)?;

        // Open WAL for appending
        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&wal_path)?;

        Ok(StorageManager {
            wal_path,
            snapshot_path,
            wal_file: Some(BufWriter::new(file)),
        })
    }

    pub fn append_wal(&mut self, entry: WalEntry) -> io::Result<()> {
        if let Some(writer) = &mut self.wal_file {
            let encoded: Vec<u8> = bincode::serialize(&entry)
                .map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;
            
            // Write length prefix (u64) - though bincode usually handles structure, 
            // for a log we need to know boundaries if we just cat them.
            // Bincode is not self-describing for streams without framing.
            // Let's use simple u32 length prefix.
            let len = encoded.len() as u32;
            writer.write_all(&len.to_le_bytes())?;
            writer.write_all(&encoded)?;
            writer.flush()?; // Ensure durability
        }
        Ok(())
    }

    pub fn save_snapshot(&self, index: &HnswIndex) -> io::Result<()> {
        let file = File::create(&self.snapshot_path)?;
        let writer = BufWriter::new(file);
        bincode::serialize_into(writer, index)
            .map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;
        Ok(())
    }

    pub fn load_snapshot(&self) -> io::Result<Option<HnswIndex>> {
        if !self.snapshot_path.exists() {
            return Ok(None);
        }
        let file = File::open(&self.snapshot_path)?;
        let reader = BufReader::new(file);
        let index: HnswIndex = bincode::deserialize_from(reader)
            .map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;
        Ok(Some(index))
    }

    // Recover by replaying WAL. 
    // This requires external logic to drive the inserts into the index.
    // We return an iterator or list of entries.
    pub fn read_wal(&self) -> io::Result<Vec<WalEntry>> {
        if !self.wal_path.exists() {
            return Ok(Vec::new());
        }
        
        let file = File::open(&self.wal_path)?;
        let mut reader = BufReader::new(file);
        let mut entries = Vec::new();
        
        use std::io::Read;
        
        loop {
            let mut len_bytes = [0u8; 4];
            match reader.read_exact(&mut len_bytes) {
                Ok(_) => {
                    let len = u32::from_le_bytes(len_bytes) as usize;
                    let mut buf = vec![0u8; len];
                    reader.read_exact(&mut buf)?;
                    
                    let entry: WalEntry = bincode::deserialize(&buf)
                        .map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;
                    entries.push(entry);
                }
                Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => break,
                Err(e) => return Err(e),
            }
        }
        
        Ok(entries)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quantizer::Quantizer;

    #[test]
    fn test_wal_write_read() {
        let dir = std::env::temp_dir().join("apexvector_test_wal");
        let _ = std::fs::remove_dir_all(&dir); // Clean up
        
        let mut sm = StorageManager::new(&dir).unwrap();
        let entry1 = WalEntry::Insert { id: 1, vector: vec![1.0, 2.0] };
        let entry2 = WalEntry::Insert { id: 2, vector: vec![3.0, 4.0] };
        
        sm.append_wal(entry1).unwrap();
        sm.append_wal(entry2).unwrap();
        
        // Re-open to read
        let sm_read = StorageManager::new(&dir).unwrap();
        let entries = sm_read.read_wal().unwrap();
        
        assert_eq!(entries.len(), 2);
        match &entries[0] {
            WalEntry::Insert { id, vector } => {
                assert_eq!(*id, 1);
                assert_eq!(vector[0], 1.0);
            }
        }
    }
}
