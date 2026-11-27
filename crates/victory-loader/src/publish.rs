//! Portable publication of derived files into a shared cache directory.
//!
//! Several copies of the runner can prepare the same title at once. Each
//! writer stages into its own file, so two of them cannot truncate each
//! other's bytes or rename a file the other already published. The final
//! rename is atomic within the directory, and the staging file is removed
//! if publication fails.

use std::fs::{File, OpenOptions};
use std::io::{ErrorKind, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

static NEXT_STAGING: AtomicU64 = AtomicU64::new(0);

/// Writes `bytes` to `path` unless it is already there, using a staging file
/// unique to this process and call.
pub fn write_if_changed(path: &Path, bytes: &[u8]) -> std::io::Result<()> {
    if std::fs::read(path).is_ok_and(|existing| existing == bytes) {
        return Ok(());
    }
    let (staging, mut file) = create_staging(path, &NEXT_STAGING)?;
    let written = file.write_all(bytes);
    // Close the handle before rename or cleanup, including on Windows.
    drop(file);
    let result = written.and_then(|()| std::fs::rename(&staging, path));
    if result.is_err() {
        let _ = std::fs::remove_file(&staging);
    }
    result
}

/// Reserve a new file without truncating leftovers from a previous process
/// that happened to have the same PID, or another writer's staging file.
fn create_staging(path: &Path, next: &AtomicU64) -> std::io::Result<(PathBuf, File)> {
    let name = path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("image");
    loop {
        let sequence = next.fetch_add(1, Ordering::Relaxed);
        let staging = path.with_file_name(format!(".{name}.{}.{sequence}.tmp", std::process::id()));
        match OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&staging)
        {
            Ok(file) => return Ok((staging, file)),
            Err(error) if error.kind() == ErrorKind::AlreadyExists => continue,
            Err(error) => return Err(error),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Two publishers staging the same destination at once both succeed,
    /// and neither leaves a staging file behind.
    #[test]
    fn concurrent_publishers_do_not_share_a_staging_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("image.bin");
        let bytes = b"a prepared image".to_vec();
        let (first, second) = (path.clone(), path.clone());
        let (first_bytes, second_bytes) = (bytes.clone(), bytes.clone());
        let (one, two) = (
            std::thread::spawn(move || write_if_changed(&first, &first_bytes)),
            std::thread::spawn(move || write_if_changed(&second, &second_bytes)),
        );
        one.join().unwrap().unwrap();
        two.join().unwrap().unwrap();
        assert_eq!(std::fs::read(&path).unwrap(), bytes);
        let leftovers: Vec<_> = std::fs::read_dir(dir.path())
            .unwrap()
            .filter_map(|entry| entry.ok())
            .filter(|entry| entry.path() != path)
            .collect();
        assert!(leftovers.is_empty(), "staging files were removed: {leftovers:?}");
    }

    #[test]
    fn stale_staging_file_is_not_truncated() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("image.bin");
        let stale = dir
            .path()
            .join(format!(".image.bin.{}.0.tmp", std::process::id()));
        std::fs::write(&stale, b"previous writer").unwrap();
        let next = AtomicU64::new(0);
        let (staging, mut file) = create_staging(&path, &next).unwrap();
        assert_ne!(staging, stale);
        file.write_all(b"new writer").unwrap();
        drop(file);
        assert_eq!(std::fs::read(&stale).unwrap(), b"previous writer");
        assert_eq!(std::fs::read(&staging).unwrap(), b"new writer");
    }

    #[test]
    fn failed_rename_removes_created_staging_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("image.bin");
        // An existing nonempty directory cannot be replaced by a file.
        std::fs::create_dir(&path).unwrap();
        std::fs::write(path.join("keep"), b"untouched").unwrap();
        assert!(write_if_changed(&path, b"bytes").is_err());
        assert_eq!(std::fs::read(path.join("keep")).unwrap(), b"untouched");
        assert_eq!(std::fs::read_dir(dir.path()).unwrap().count(), 1);
    }

    #[test]
    fn a_failed_publication_removes_its_staging_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("missing-parent").join("image.bin");
        assert!(write_if_changed(&path, b"bytes").is_err());
        let leftovers: Vec<_> = std::fs::read_dir(dir.path())
            .unwrap()
            .filter_map(|entry| entry.ok())
            .collect();
        assert!(leftovers.is_empty(), "no staging file survives an error");
    }
}
