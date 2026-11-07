//! `agentero discover *` — query-first paper discovery.
//!
//! Headless, vault-free: fetch arXiv candidates for a topic/category/date
//! window and rank them with a deterministic lexical scorer. The pipeline
//! (and any external agent) can consume the JSON shortlist directly.
//!
//! Cross-round memory: a plain JSONL "pushed" ledger records every surfaced
//! shortlist, so the next round (same or different keywords, even a journal
//! RSS entry recorded from another source) does not re-push the same paper.
//! The ledger is on by default — `<vault>/.agentero/discovery/seen.jsonl`
//! when a vault resolves, else the XDG data dir; `--seen FILE` overrides the
//! path, `--no-seen` disables it.

use crate::error::CliError;
use crate::output::to_value;
use crate::resolve::{resolve_vault, GlobalOpts};
use crate::style::{format_table, truncate_chars};
use agentero_core::features::paper::discovery::discover::{
    discover_arxiv, known_ids, DiscoverQuery, Exclusion, DEFAULT_MAX_CANDIDATES, DEFAULT_TOP,
};
use agentero_core::features::paper::discovery::embeddings::EmbeddingConfig;
use agentero_core::features::paper::discovery::seen::{self, SeenEntry, SeenLedger};
use clap::Subcommand;
use serde::Serialize;
use serde_json::{json, Value};
use std::collections::HashSet;
use std::fs;
use std::path::{Path, PathBuf};

#[derive(Debug, Subcommand)]
pub enum DiscoverCmd {
    /// Rank the day's arXiv submissions for keywords / categories.
    ///
    /// Candidates come from the arXiv Atom API (newest submissions first) and
    /// are scored lexically: a title hit weighs more than an abstract hit.
    /// Provide at least one --keyword or --category.
    Arxiv {
        /// Topic term or phrase (repeatable). e.g. -k agent -k "world model".
        #[arg(short = 'k', long = "keyword", value_name = "TERM")]
        keywords: Vec<String>,
        /// arXiv category (repeatable), e.g. -c cs.AI.
        #[arg(short = 'c', long = "category", value_name = "CAT")]
        categories: Vec<String>,
        /// Only papers submitted on/after this date (YYYY-MM-DD).
        #[arg(long = "since", value_name = "DATE")]
        since: Option<String>,
        /// Only papers submitted on/before this date (YYYY-MM-DD).
        #[arg(long = "until", value_name = "DATE")]
        until: Option<String>,
        /// Shortlist size.
        #[arg(long = "top", value_name = "N", default_value_t = DEFAULT_TOP)]
        top: usize,
        /// Max candidates to fetch before ranking.
        #[arg(long = "max-candidates", value_name = "N", default_value_t = DEFAULT_MAX_CANDIDATES)]
        max_candidates: usize,
        /// Write the full shortlist JSON to this file (creation is idempotent).
        #[arg(long = "out", value_name = "FILE", value_hint = clap::ValueHint::FilePath)]
        out: Option<PathBuf>,
        /// Do not drop papers already in the library catalog.
        #[arg(long = "no-dedup")]
        no_dedup: bool,
        /// Do not consult or update the pushed ledger (cross-round dedup).
        #[arg(long = "no-seen")]
        no_seen: bool,
        /// Pushed-ledger file. Default: <vault>/.agentero/discovery/seen.jsonl,
        /// else the XDG data dir. JSONL, one entry per pushed item.
        #[arg(long = "seen", value_name = "FILE", value_hint = clap::ValueHint::FilePath)]
        seen: Option<PathBuf>,
        /// OpenAI-compatible embeddings base URL; enables optional semantic
        /// ranking. Key is read from AGENTERO_EMBEDDING_API_KEY.
        #[arg(long = "embed-base", value_name = "URL")]
        embed_base: Option<String>,
        /// Embeddings model id (required with --embed-base).
        #[arg(long = "embed-model", value_name = "MODEL")]
        embed_model: Option<String>,
        /// Cosine weight added to the lexical score (default 1.0).
        #[arg(long = "semantic-weight", value_name = "F")]
        semantic_weight: Option<f32>,
    },
    /// Manage the pushed ledger — entries earlier discovery rounds already
    /// surfaced, deduped against on the next round.
    Seen {
        #[command(subcommand)]
        action: SeenAction,
    },
}

#[derive(Debug, Subcommand)]
pub enum SeenAction {
    /// List ledger entries (newest first).
    List {
        #[arg(long, default_value_t = 50)]
        limit: usize,
        #[command(flatten)]
        path: SeenPath,
    },
    /// Record an entry, e.g. a journal RSS item the radar pushed — so the
    /// next round drops its arXiv preprint (matched by DOI or title).
    Add {
        /// Entry title (required; always available, even without ids).
        #[arg(short = 't', long)]
        title: String,
        #[arg(long = "arxiv-id")]
        arxiv_id: Option<String>,
        #[arg(long)]
        doi: Option<String>,
        #[arg(long, default_value = "feed")]
        source: String,
        #[command(flatten)]
        path: SeenPath,
    },
    /// Remove entries matching an arXiv id, DOI or title (normalized).
    Forget {
        #[arg(value_name = "MATCH")]
        needle: String,
        #[command(flatten)]
        path: SeenPath,
    },
    /// Remove all entries (destructive; requires --yes).
    Clear {
        #[command(flatten)]
        path: SeenPath,
    },
}

/// `--seen FILE` override shared by the ledger subcommands.
#[derive(Debug, Clone, clap::Args)]
pub struct SeenPath {
    /// Pushed-ledger file (default: <vault>/.agentero/discovery/seen.jsonl,
    /// else the XDG data dir).
    #[arg(long = "seen", value_name = "FILE", value_hint = clap::ValueHint::FilePath)]
    seen: Option<PathBuf>,
}

/// Resolve the ledger path: `--seen` wins, then the vault default, then the
/// XDG data dir (headless radar boxes usually have no vault). An explicit
/// `--vault` that cannot resolve is still an error.
fn seen_path(globals: &GlobalOpts, explicit: Option<&Path>) -> Result<PathBuf, CliError> {
    if let Some(path) = explicit {
        return Ok(path.to_path_buf());
    }
    match resolve_vault(globals) {
        Ok(vault) => Ok(seen::default_ledger_path(Some(&vault))),
        Err(err) => {
            if globals.vault_flag.is_some() {
                return Err(err);
            }
            Ok(seen::default_ledger_path(None))
        }
    }
}

pub async fn run(cmd: DiscoverCmd, globals: &GlobalOpts) -> Result<Value, CliError> {
    match cmd {
        DiscoverCmd::Arxiv {
            keywords,
            categories,
            since,
            until,
            top,
            max_candidates,
            out,
            no_dedup,
            no_seen,
            seen,
            embed_base,
            embed_model,
            semantic_weight,
        } => {
            run_arxiv(
                globals,
                ArxivInputs {
                    keywords,
                    categories,
                    since,
                    until,
                    top,
                    max_candidates,
                    out,
                    no_dedup,
                    no_seen,
                    seen,
                    embed_base,
                    embed_model,
                    semantic_weight,
                },
            )
            .await
        }
        DiscoverCmd::Seen { action } => run_seen(globals, action).await,
    }
}

struct ArxivInputs {
    keywords: Vec<String>,
    categories: Vec<String>,
    since: Option<String>,
    until: Option<String>,
    top: usize,
    max_candidates: usize,
    out: Option<PathBuf>,
    no_dedup: bool,
    no_seen: bool,
    seen: Option<PathBuf>,
    embed_base: Option<String>,
    embed_model: Option<String>,
    semantic_weight: Option<f32>,
}

async fn run_arxiv(globals: &GlobalOpts, inputs: ArxivInputs) -> Result<Value, CliError> {
    let query = DiscoverQuery {
        keywords: inputs.keywords,
        categories: inputs.categories,
        since: inputs.since,
        until: inputs.until,
        top: Some(inputs.top),
        max_candidates: Some(inputs.max_candidates),
        semantic_weight: inputs.semantic_weight,
    };

    // Pushed ledger: entries earlier rounds already surfaced.
    let ledger_path = if inputs.no_seen {
        None
    } else {
        Some(seen_path(globals, inputs.seen.as_deref())?)
    };
    let mut ledger = match &ledger_path {
        Some(path) => Some(SeenLedger::load(path)?),
        None => None,
    };

    let exclusion = Exclusion {
        known: if inputs.no_dedup {
            HashSet::new()
        } else {
            known_catalog_ids(globals)?
        },
        seen: ledger
            .as_ref()
            .map(|l| l.keys().clone())
            .unwrap_or_default(),
    };
    let embedding = embedding_config(inputs.embed_base.as_deref(), inputs.embed_model.as_deref());
    let data = discover_arxiv(&query, &exclusion, embedding.as_ref()).await?;

    // Record the surfaced shortlist so the next round drops it.
    let recorded = match (&mut ledger, ledger_path.as_ref()) {
        (Some(ledger), Some(path)) if !data.items.is_empty() => {
            let added = ledger.add_items(
                data.items.iter().map(|item| {
                    (
                        item.title.as_str(),
                        if item.arxiv_id.is_empty() {
                            None
                        } else {
                            Some(item.arxiv_id.as_str())
                        },
                        item.doi.as_deref(),
                    )
                }),
                "arxiv",
            );
            ledger.save(path)?;
            added
        }
        _ => 0,
    };

    if let Some(path) = &inputs.out {
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                fs::create_dir_all(parent)?;
            }
        }
        fs::write(path, serde_json::to_string_pretty(&data)?)?;
    }

    let rows: Vec<Vec<String>> = data
        .items
        .iter()
        .enumerate()
        .map(|(i, item)| {
            vec![
                (i + 1).to_string(),
                format!("{:.1}", item.score),
                item.arxiv_id.clone(),
                truncate_chars(&item.title, 72),
            ]
        })
        .collect();
    let mut lines = if data.items.is_empty() {
        vec![globals.style.dim("(no candidates)")]
    } else {
        format_table(globals.style, &["#", "SCORE", "arXiv ID", "TITLE"], &rows)
    };
    lines.push(globals.style.dim(&format!(
        "scanned {} · excluded {} (library) · seen {} (pushed) · shortlisted {} · recorded {recorded}",
        data.candidates_scanned,
        data.excluded,
        data.seen_excluded,
        data.items.len()
    )));
    if let Some(err) = &data.semantic_error {
        lines.push(
            globals
                .style
                .dim(&format!("semantic ranking skipped: {err}")),
        );
    }

    let mut out = to_value(&data)?;
    if let Some(obj) = out.as_object_mut() {
        obj.insert("lines".into(), json!(lines));
        obj.insert("recorded".into(), json!(recorded));
        if let Some(path) = &ledger_path {
            obj.insert("seenFile".into(), json!(path.to_string_lossy().to_string()));
        }
    }
    Ok(out)
}

/// Library identity keys (arXiv ids + DOIs) to drop from the shortlist.
///
/// Discovery is vault-free, so a resolvable vault enables novelty filtering
/// but its absence is not an error — unless the user explicitly passed
/// `--vault`.
fn known_catalog_ids(globals: &GlobalOpts) -> Result<HashSet<String>, CliError> {
    match resolve_vault(globals) {
        Ok(vault) => Ok(known_ids(&vault)?),
        Err(err) => {
            if globals.vault_flag.is_some() {
                Err(err)
            } else {
                Ok(HashSet::new())
            }
        }
    }
}

/// Build an embedding config when both a base URL and a model are given.
/// The key is read from `AGENTERO_EMBEDDING_API_KEY` (never argv, to avoid
/// leaking into the process list).
fn embedding_config(base: Option<&str>, model: Option<&str>) -> Option<EmbeddingConfig> {
    let base = base?.trim();
    let model = model?.trim();
    if base.is_empty() || model.is_empty() {
        return None;
    }
    let api_key = std::env::var("AGENTERO_EMBEDDING_API_KEY")
        .ok()
        .map(|k| k.trim().to_string())
        .filter(|k| !k.is_empty());
    Some(EmbeddingConfig {
        base_url: base.to_string(),
        api_key,
        model: model.to_string(),
    })
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct SeenListOut {
    file: String,
    count: usize,
    entries: Vec<SeenEntry>,
}

async fn run_seen(globals: &GlobalOpts, action: SeenAction) -> Result<Value, CliError> {
    match action {
        SeenAction::List { limit, path } => {
            let file = seen_path(globals, path.seen.as_deref())?;
            let ledger = SeenLedger::load(&file)?;
            let mut entries: Vec<SeenEntry> = ledger.entries().to_vec();
            entries.reverse(); // appended in record order → newest first
            entries.truncate(limit);
            let out = SeenListOut {
                file: file.to_string_lossy().to_string(),
                count: ledger.len(),
                entries,
            };
            let mut value = to_value(&out)?;
            if let Some(obj) = value.as_object_mut() {
                let lines: Vec<String> = out
                    .entries
                    .iter()
                    .map(|e| {
                        let id = e
                            .arxiv_id
                            .clone()
                            .or_else(|| e.doi.clone())
                            .unwrap_or_else(|| e.source.clone().unwrap_or_default());
                        format!(
                            "{}  {}  {}",
                            e.at,
                            truncate_chars(&id, 24),
                            truncate_chars(&e.title, 72)
                        )
                    })
                    .collect();
                obj.insert(
                    "lines".into(),
                    json!(if lines.is_empty() {
                        vec![globals.style.dim("(ledger is empty)")]
                    } else {
                        lines
                    }),
                );
            }
            Ok(value)
        }
        SeenAction::Add {
            title,
            arxiv_id,
            doi,
            source,
            path,
        } => {
            let file = seen_path(globals, path.seen.as_deref())?;
            let mut ledger = SeenLedger::load(&file)?;
            let entry = SeenEntry {
                title: title.trim().to_string(),
                arxiv_id: arxiv_id.filter(|s| !s.trim().is_empty()),
                doi: doi.filter(|s| !s.trim().is_empty()),
                source: Some(source.trim().to_string()),
                at: agentero_core::time::now_rfc3339_millis(),
            };
            let added = ledger.add([entry]);
            ledger.save(&file)?;
            Ok(json!({
                "file": file.to_string_lossy().to_string(),
                "added": added,
                "count": ledger.len(),
            }))
        }
        SeenAction::Forget { needle, path } => {
            let file = seen_path(globals, path.seen.as_deref())?;
            let mut ledger = SeenLedger::load(&file)?;
            let removed = ledger.forget(&needle);
            if removed > 0 {
                ledger.save(&file)?;
            }
            Ok(json!({
                "file": file.to_string_lossy().to_string(),
                "removed": removed,
                "count": ledger.len(),
            }))
        }
        SeenAction::Clear { path } => {
            if !globals.yes {
                return Err(CliError::usage(
                    "`discover seen clear` removes every entry; pass --yes to confirm",
                ));
            }
            let file = seen_path(globals, path.seen.as_deref())?;
            let mut ledger = SeenLedger::load(&file)?;
            let removed = ledger.len();
            ledger.clear();
            ledger.save(&file)?;
            Ok(json!({
                "file": file.to_string_lossy().to_string(),
                "removed": removed,
                "count": ledger.len(),
            }))
        }
    }
}
