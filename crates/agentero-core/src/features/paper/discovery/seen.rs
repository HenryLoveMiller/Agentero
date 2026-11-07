//! The "pushed" ledger: entries earlier discovery rounds already surfaced.
//!
//! Catalog-based novelty filtering only knows what is in the library. A
//! radar that runs several queries per day and pushes each shortlist
//! elsewhere (chat, digest) has no memory across rounds: the next query
//! re-scores and re-pushes papers the previous query already delivered
//! under a different keyword. The ledger closes that gap — every surfaced
//! shortlist is appended as one JSON line per item, and the next run drops
//! any candidate that matches a stored entry by arXiv id, DOI or
//! normalized title. Because matching is identity-key based, a journal RSS
//! entry (DOI/title only) recorded here also drops its arXiv preprint, and
//! vice versa.
//!
//! Storage is a plain JSONL file so any radar script can read, append or
//! reset it without the app:
//!
//! ```text
//! {"title":"…","arxivId":"2401.12345","doi":"10.1/x","source":"arxiv","at":"2026-11-07T…Z"}
//! ```

use crate::error::AppError;
use crate::features::paper::discovery::identity::{self, tagged_keys};
use crate::paths;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::fs;
use std::path::{Path, PathBuf};

/// One pushed entry. `title` is required (it is always present, even for
/// sources with no machine ids); the rest are kept when the source has them.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SeenEntry {
    pub title: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub arxiv_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub doi: Option<String>,
    /// Where the item came from (`"arxiv"`, `"feed"`, …) — informational.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source: Option<String>,
    /// RFC 3339 millis stamp of when the entry was recorded.
    pub at: String,
}

impl SeenEntry {
    /// Identity keys this entry occupies; empty for entries with no usable
    /// identifier at all (they are skipped on add, so they never match).
    pub fn keys(&self) -> Vec<String> {
        tagged_keys(self.arxiv_id.as_deref(), self.doi.as_deref(), &self.title)
    }
}

/// In-memory view of the JSONL ledger: entries plus the precomputed key set
/// used for O(1) matching.
#[derive(Debug, Clone, Default)]
pub struct SeenLedger {
    entries: Vec<SeenEntry>,
    keys: HashSet<String>,
}

impl SeenLedger {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn entries(&self) -> &[SeenEntry] {
        &self.entries
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// All identity keys occupied by stored entries.
    pub fn keys(&self) -> &HashSet<String> {
        &self.keys
    }

    /// Load the ledger at `path`. A missing file is an empty ledger;
    /// malformed lines are skipped (with a warning) so one bad append can
    /// never brick a long-running radar.
    pub fn load(path: &Path) -> Result<SeenLedger, AppError> {
        let content = match fs::read_to_string(path) {
            Ok(content) => content,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(Self::default()),
            Err(e) => return Err(e.into()),
        };
        let mut ledger = Self::default();
        for (index, line) in content.lines().enumerate() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            match serde_json::from_str::<SeenEntry>(line) {
                Ok(entry) => {
                    ledger.add_one(entry);
                }
                Err(err) => log::warn!(
                    target: "agentero::discover::seen",
                    "ledger {}: skipping line {}: {err}",
                    path.display(),
                    index + 1
                ),
            }
        }
        Ok(ledger)
    }

    /// Record items as seen, one entry per `(title, arxiv_id, doi)` triple.
    /// Works for both [`crate::features::paper::scholar_api::ApiPaper`]
    /// candidates and `DiscoverItem` shortlists. Returns the number of
    /// entries actually added (duplicates against existing keys are skipped).
    pub fn add_items<'a>(
        &mut self,
        items: impl IntoIterator<Item = (&'a str, Option<&'a str>, Option<&'a str>)>,
        source: &str,
    ) -> usize {
        let now = crate::time::now_rfc3339_millis();
        let entries = items.into_iter().map(|(title, arxiv_id, doi)| SeenEntry {
            title: title.trim().to_string(),
            arxiv_id: arxiv_id
                .map(str::trim)
                .filter(|s| !s.is_empty())
                .map(str::to_string),
            doi: doi
                .map(str::trim)
                .filter(|s| !s.is_empty())
                .map(str::to_string),
            source: Some(source.to_string()),
            at: now.clone(),
        });
        self.add(entries)
    }

    /// Append entries that do not already occupy a stored key. Returns the
    /// number of entries actually added.
    pub fn add(&mut self, entries: impl IntoIterator<Item = SeenEntry>) -> usize {
        let mut added = 0;
        for entry in entries {
            if self.add_one(entry) {
                added += 1;
            }
        }
        added
    }

    /// Remove every entry that shares a key with `needle`. The needle may be
    /// a raw arXiv id, a DOI or a title; each is normalized, so
    /// `forget "arXiv:2401.12345v2"` and `forget "2401.12345"` remove the
    /// same entry. Returns the number removed.
    pub fn forget(&mut self, needle: &str) -> usize {
        let needle = needle.trim();
        if needle.is_empty() {
            return 0;
        }
        let mut needles: HashSet<String> = HashSet::new();
        if let Some(k) = identity::normalize_arxiv_id(needle) {
            needles.insert(format!("arxiv:{k}"));
        }
        if let Some(k) = identity::normalize_doi(needle) {
            needles.insert(format!("doi:{k}"));
        }
        if let Some(k) = identity::normalize_title(needle) {
            needles.insert(format!("title:{k}"));
        }
        let before = self.entries.len();
        self.entries
            .retain(|entry| entry.keys().iter().all(|k| !needles.contains(k)));
        let removed = before - self.entries.len();
        if removed > 0 {
            self.rebuild_keys();
        }
        removed
    }

    pub fn clear(&mut self) {
        self.entries.clear();
        self.keys.clear();
    }

    /// Write the ledger atomically (tmp file + rename), creating parent
    /// directories.
    pub fn save(&self, path: &Path) -> Result<(), AppError> {
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                fs::create_dir_all(parent)?;
            }
        }
        let mut buf = String::new();
        for entry in &self.entries {
            buf.push_str(&serde_json::to_string(entry)?);
            buf.push('\n');
        }
        let tmp = path.with_extension("tmp");
        fs::write(&tmp, buf)?;
        fs::rename(&tmp, path)?;
        Ok(())
    }

    fn add_one(&mut self, entry: SeenEntry) -> bool {
        let keys = entry.keys();
        if keys.is_empty() || keys.iter().any(|k| self.keys.contains(k)) {
            return false;
        }
        for key in keys {
            self.keys.insert(key);
        }
        self.entries.push(entry);
        true
    }

    fn rebuild_keys(&mut self) {
        self.keys.clear();
        for entry in &self.entries {
            for key in entry.keys() {
                self.keys.insert(key);
            }
        }
    }
}

/// Default ledger location: `<vault>/.agentero/discovery/seen.jsonl` when a
/// vault is resolvable, else `$XDG_DATA_HOME/agentero/discovery/seen.jsonl`
/// (headless radar boxes usually run without a vault).
pub fn default_ledger_path(vault: Option<&Path>) -> PathBuf {
    match vault {
        Some(vault) => vault.join(".agentero").join("discovery").join("seen.jsonl"),
        None => paths::agentero_data_dir()
            .join("discovery")
            .join("seen.jsonl"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn entry(title: &str, arxiv_id: Option<&str>, doi: Option<&str>) -> SeenEntry {
        SeenEntry {
            title: title.to_string(),
            arxiv_id: arxiv_id.map(str::to_string),
            doi: doi.map(str::to_string),
            source: Some("test".into()),
            at: "2026-11-07T00:00:00.000Z".into(),
        }
    }

    fn temp_dir(tag: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!(
            "agentero-seen-{tag}-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ))
    }

    #[test]
    fn load_missing_file_is_empty_ledger() {
        let path = temp_dir("missing").join("seen.jsonl");
        let ledger = SeenLedger::load(&path).unwrap();
        assert!(ledger.is_empty());
    }

    #[test]
    fn save_load_roundtrip_and_bad_lines_skip() {
        let dir = temp_dir("roundtrip");
        let path = dir.join("seen.jsonl");
        {
            let mut ledger = SeenLedger::new();
            ledger.add([
                entry("A Paper", Some("2401.00001"), None),
                entry("B Paper", None, Some("10.1/b")),
            ]);
            ledger.save(&path).unwrap();
        }
        // Append a malformed line the way a broken script would.
        fs::write(&path, fs::read_to_string(&path).unwrap() + "not json\n").unwrap();
        let ledger = SeenLedger::load(&path).unwrap();
        assert_eq!(ledger.len(), 2);
        assert!(ledger.keys().contains("arxiv:2401.00001"));
        assert!(ledger.keys().contains("doi:10.1/b"));
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn add_is_idempotent_on_shared_keys() {
        let mut ledger = SeenLedger::new();
        assert_eq!(
            ledger.add([entry("Same Title", Some("2401.00001"), None)]),
            1
        );
        // Same arXiv id, even with a different title: one paper.
        assert_eq!(
            ledger.add([entry("Different Title", Some("arXiv:2401.00001v2"), None)]),
            0
        );
        // Same normalized title, different casing/punctuation: one paper.
        assert_eq!(ledger.add([entry("same  title!!!", None, None)]), 0);
        // A genuinely new paper (shares nothing).
        assert_eq!(ledger.add([entry("Other", None, Some("10.2/c"))]), 1);
        assert_eq!(ledger.len(), 2);
        // An entry with no usable identifier is never stored.
        assert_eq!(ledger.add([entry("   ", None, None)]), 0);
    }

    #[test]
    fn add_items_records_shortlist() {
        let mut ledger = SeenLedger::new();
        let items = [
            ("One", Some("2401.00001"), None),
            ("Two", Some("2401.00002"), None),
        ];
        assert_eq!(ledger.add_items(items, "arxiv"), 2);
        // Re-recording the same shortlist adds nothing.
        assert_eq!(ledger.add_items(items, "arxiv"), 0);
    }

    #[test]
    fn forget_by_id_doi_or_title() {
        let mut ledger = SeenLedger::new();
        ledger.add([
            entry("A Paper", Some("2401.00001"), None),
            entry("B Paper", None, Some("10.1/b")),
            entry("C Paper", None, None),
        ]);
        assert_eq!(ledger.forget("arXiv:2401.00001v3"), 1);
        assert_eq!(ledger.forget("https://doi.org/10.1/B"), 1);
        assert_eq!(ledger.forget("c  paper"), 1);
        assert!(ledger.is_empty());
        assert_eq!(ledger.forget("nothing here"), 0);
    }

    #[test]
    fn default_path_prefers_vault() {
        let vault = Path::new("/tmp/v");
        assert_eq!(
            default_ledger_path(Some(vault)),
            PathBuf::from("/tmp/v/.agentero/discovery/seen.jsonl")
        );
        let fallback = default_ledger_path(None);
        assert!(
            fallback
                .to_string_lossy()
                .ends_with("agentero/discovery/seen.jsonl"),
            "unexpected fallback path: {fallback:?}"
        );
    }
}
