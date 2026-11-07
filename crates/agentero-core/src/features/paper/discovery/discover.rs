//! Query-first arXiv discovery.
//!
//! Given a topic (keywords), a category set, and/or a submission-date window,
//! fetch candidates from the arXiv Atom API and rank them with a
//! **deterministic lexical scorer** — no embedding endpoint, no API key, no
//! catalog. This is the headless "抓当天 arXiv + 关键词排序" primitive that the
//! CLI / Agent can drive; the desktop plaza panel's library-profile ranking is
//! a separate concern.
//!
//! The score is explainable: every item carries the terms that matched and how
//! much each contributed (title hits weigh more than abstract hits).

use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::path::Path;

use crate::error::AppError;
use crate::features::paper::discovery::embeddings::{self, EmbeddingConfig};
use crate::features::paper::discovery::identity::{self, tagged_keys};
use crate::features::scholar_api::sources::arxiv;
use crate::features::scholar_api::ApiPaper;

/// Title term weight — a title hit is a stronger topic signal than an abstract
/// hit, mirroring the classic keyword shortlist heuristic.
const TITLE_WEIGHT: f32 = 3.0;
/// Abstract term weight.
const ABSTRACT_WEIGHT: f32 = 1.0;
/// Candidates scanned when the caller does not cap them.
pub const DEFAULT_MAX_CANDIDATES: usize = 100;
/// Shortlist size when the caller does not ask for a specific one.
pub const DEFAULT_TOP: usize = 8;
/// Cosine weight added on top of the lexical score when semantic ranking runs.
pub const DEFAULT_SEMANTIC_WEIGHT: f32 = 1.0;
/// Chars of a candidate sent for embedding (providers cap tokens per input).
const MAX_EMBED_CHARS: usize = 4_000;

/// A discovery request. At least one keyword or category is required; a date
/// window alone would scan an unbounded slice of arXiv.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DiscoverQuery {
    /// Topic terms / phrases (repeatable).
    #[serde(default)]
    pub keywords: Vec<String>,
    /// arXiv categories, e.g. `cs.AI`.
    #[serde(default)]
    pub categories: Vec<String>,
    /// Only papers submitted on/after this date (`YYYY-MM-DD`).
    #[serde(default)]
    pub since: Option<String>,
    /// Only papers submitted on/before this date (`YYYY-MM-DD`).
    #[serde(default)]
    pub until: Option<String>,
    /// Shortlist size.
    #[serde(default)]
    pub top: Option<usize>,
    /// Max candidates fetched before ranking.
    #[serde(default)]
    pub max_candidates: Option<usize>,
    /// Cosine weight added on top of the lexical score when an embedding
    /// endpoint is supplied (default [`DEFAULT_SEMANTIC_WEIGHT`]); ignored when
    /// no endpoint is passed.
    #[serde(default)]
    pub semantic_weight: Option<f32>,
}

/// One term's contribution to an item's score (for explainability).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ScoreTerm {
    pub term: String,
    /// `"title"` or `"abstract"`.
    pub field: String,
    pub count: u32,
    pub weight: f32,
    pub contribution: f32,
}

/// A ranked discovery candidate.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DiscoverItem {
    pub arxiv_id: String,
    /// DOI when the source provides one (arXiv Atom usually does not).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub doi: Option<String>,
    pub title: String,
    #[serde(rename = "abstract")]
    pub abstract_text: String,
    pub url: String,
    pub pdf_url: Option<String>,
    pub published_at: Option<String>,
    pub score: f32,
    /// Non-empty matchers, highest-contribution first is not guaranteed.
    pub matches: Vec<ScoreTerm>,
}

/// Candidate keys to drop before ranking, partitioned by origin so the
/// result can report the two filters separately.
///
/// Keys are tagged identity keys (see [`crate::features::paper::discovery::identity`]):
/// `arxiv:…`, `doi:…` or `title:…`. A candidate is dropped when *any* of its
/// own keys is present in the corresponding set.
#[derive(Debug, Clone, Default)]
pub struct Exclusion {
    /// Identity keys of papers already in the library catalog.
    pub known: HashSet<String>,
    /// Identity keys of entries a previous round already pushed (ledger).
    pub seen: HashSet<String>,
}

impl Exclusion {
    pub fn is_empty(&self) -> bool {
        self.known.is_empty() && self.seen.is_empty()
    }
}

/// Result of one discovery run.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DiscoverResult {
    pub source: String,
    /// Echo of the request (with resolved defaults omitted).
    pub query: DiscoverQuery,
    /// The raw arXiv `search_query` expression, for transparency/replay.
    pub search_query: String,
    /// Candidates fetched before novelty filtering.
    pub candidates_scanned: usize,
    /// Candidates dropped because they are already known (in the library).
    pub excluded: usize,
    /// Candidates dropped because a previous round already pushed them
    /// (the "pushed" ledger).
    #[serde(default)]
    pub seen_excluded: usize,
    pub computed_at: String,
    /// Set when semantic ranking was requested but the endpoint failed; the
    /// result then degrades to lexical-only instead of erroring.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub semantic_error: Option<String>,
    pub items: Vec<DiscoverItem>,
}

/// Fetch + rank arXiv candidates for `query`.
///
/// `exclude` drops candidates already known to the caller *before* ranking,
/// so the shortlist fills with fresh papers instead of being eaten by a
/// truncated `top`: `known` for the library catalog, `seen` for entries a
/// previous round already pushed (the ledger). Matching is identity-key
/// based (arXiv id, DOI or normalized title), so a journal RSS entry
/// recorded in the ledger also drops its arXiv preprint. Pass
/// [`Exclusion::default()`] for a pure network discovery run.
///
/// `embedding`, when supplied, adds an optional cosine-similarity term to the
/// lexical score (see [`DiscoverQuery::semantic_weight`]); if that call fails
/// the run degrades to lexical-only and records `semantic_error`.
///
/// Network-only otherwise: does not touch the Vault, catalog or ledger — the
/// caller owns loading and updating them.
pub async fn discover_arxiv(
    query: &DiscoverQuery,
    exclude: &Exclusion,
    embedding: Option<&EmbeddingConfig>,
) -> Result<DiscoverResult, AppError> {
    let search_query = build_search_query(query)?;
    let max_candidates = query
        .max_candidates
        .unwrap_or(DEFAULT_MAX_CANDIDATES)
        .clamp(1, 200);
    let top = query.top.unwrap_or(DEFAULT_TOP).clamp(1, max_candidates);

    let papers = arxiv::query_atom(&search_query, 0, max_candidates).await?;
    let candidates_scanned = papers.len();

    let (kept, excluded, seen_excluded) = drop_known(papers, exclude);
    let terms = normalize_terms(&query.keywords);
    let mut items = score_lexical(kept, &terms);

    let mut semantic_error = None;
    if let Some(config) = embedding {
        if !terms.is_empty() && !items.is_empty() {
            let weight = query.semantic_weight.unwrap_or(DEFAULT_SEMANTIC_WEIGHT);
            if let Err(err) = apply_semantic(&mut items, &terms, config, weight).await {
                log::warn!(target: "agentero::discover", "semantic ranking skipped: {err}");
                semantic_error = Some(err.to_string());
            }
        }
    }

    let items = finalize(items, top);

    Ok(DiscoverResult {
        source: "arxiv".to_string(),
        query: query.clone(),
        search_query,
        candidates_scanned,
        excluded,
        seen_excluded,
        computed_at: crate::time::now_rfc3339_millis(),
        semantic_error,
        items,
    })
}

/// Add `weight * cosine(query, candidate)` to each item and record it as a
/// `semantic` score term. Degrades to lexical on any endpoint error.
async fn apply_semantic(
    items: &mut [DiscoverItem],
    terms: &[String],
    config: &EmbeddingConfig,
    weight: f32,
) -> Result<(), AppError> {
    let mut texts: Vec<String> = Vec::with_capacity(items.len() + 1);
    texts.push(terms.join(" "));
    for item in items.iter() {
        texts.push(embed_input(&item.title, &item.abstract_text));
    }

    let mut vectors = embeddings::embed_texts(config, &texts).await?;
    if vectors.len() != texts.len() {
        return Err(AppError::message(
            "embeddings returned the wrong number of vectors",
        ));
    }
    for vector in vectors.iter_mut() {
        embeddings::normalize(vector);
    }
    let query_vector = vectors.remove(0);
    apply_semantic_vectors(items, &query_vector, &vectors, weight);
    Ok(())
}

/// Add `weight * cosine(query, candidate)` to each item and record it as a
/// `semantic` score term. Pure so the blend is unit-testable.
fn apply_semantic_vectors(
    items: &mut [DiscoverItem],
    query_vector: &[f32],
    vectors: &[Vec<f32>],
    weight: f32,
) {
    for (item, candidate) in items.iter_mut().zip(vectors.iter()) {
        let contribution = if query_vector.len() == candidate.len() {
            weight * embeddings::dot(query_vector, candidate)
        } else {
            0.0
        };
        item.score += contribution;
        item.matches.push(ScoreTerm {
            term: "<semantic>".to_string(),
            field: "semantic".to_string(),
            count: 0,
            weight,
            contribution,
        });
    }
}

fn embed_input(title: &str, abstract_text: &str) -> String {
    let joined = format!("{}\n\n{}", title.trim(), abstract_text.trim());
    joined.chars().take(MAX_EMBED_CHARS).collect()
}

/// Split candidates into "kept", "already in the library" and "already
/// pushed" (dropped). A candidate is dropped when any of its identity keys
/// (arXiv id, DOI, normalized title) is in the corresponding exclusion set;
/// the library filter wins the attribution. Candidates whose keys are all
/// unnormalizable are always kept — novelty cannot be judged.
fn drop_known(papers: Vec<ApiPaper>, exclude: &Exclusion) -> (Vec<ApiPaper>, usize, usize) {
    if exclude.is_empty() {
        return (papers, 0, 0);
    }
    let mut kept = Vec::with_capacity(papers.len());
    let mut excluded = 0usize;
    let mut seen_excluded = 0usize;
    for paper in papers {
        let keys = tagged_keys(
            paper.identifiers.arxiv_id.as_deref(),
            paper.identifiers.doi.as_deref(),
            &paper.title,
        );
        if keys.iter().any(|key| exclude.known.contains(key)) {
            excluded += 1;
            continue;
        }
        if keys.iter().any(|key| exclude.seen.contains(key)) {
            seen_excluded += 1;
            continue;
        }
        kept.push(paper);
    }
    (kept, excluded, seen_excluded)
}

/// Identity keys (arXiv ids + DOIs) of papers already in the vault's catalog,
/// for novelty filtering.
///
/// DOIs are included so a journal version in the library also matches its
/// arXiv preprint candidate (and vice versa). Titles are deliberately not
/// included: a permanent library should never hide a paper on a fuzzy match.
///
/// Returns an empty set when the catalog has no arXiv/DOI papers.
pub fn known_ids(vault_root: &Path) -> Result<HashSet<String>, AppError> {
    let papers = crate::features::paper::catalog::papers::list_all_unique_by_id(vault_root)?;
    let mut keys = HashSet::new();
    for paper in papers {
        if let Some(id) = paper
            .arxiv_id
            .as_deref()
            .and_then(identity::normalize_arxiv_id)
        {
            keys.insert(format!("arxiv:{id}"));
        }
        if let Some(doi) = paper.doi.as_deref().and_then(identity::normalize_doi) {
            keys.insert(format!("doi:{doi}"));
        }
    }
    Ok(keys)
}

/// Compose the arXiv `search_query` expression from keywords / categories /
/// submission-date bounds. Pure so it is unit-testable without the network.
fn build_search_query(query: &DiscoverQuery) -> Result<String, AppError> {
    let mut clauses: Vec<String> = Vec::new();

    for keyword in &query.keywords {
        let keyword = keyword.trim();
        if keyword.is_empty() {
            continue;
        }
        if keyword.chars().any(char::is_whitespace) {
            // Multi-word phrase → quoted so arXiv treats it as one unit.
            clauses.push(format!("all:\"{}\"", keyword.replace('"', " ")));
        } else {
            clauses.push(format!("all:{keyword}"));
        }
    }

    for category in &query.categories {
        let category = category.trim();
        if !category.is_empty() {
            clauses.push(format!("cat:{category}"));
        }
    }

    let from = query
        .since
        .as_deref()
        .map(|d| date_bound(d, false))
        .transpose()?;
    let to = query
        .until
        .as_deref()
        .map(|d| date_bound(d, true))
        .transpose()?;
    match (from, to) {
        (Some(from), Some(to)) => clauses.push(format!("submittedDate:[{from} TO {to}]")),
        (Some(from), None) => clauses.push(format!("submittedDate:[{from} TO 999912312359]")),
        (None, Some(to)) => clauses.push(format!("submittedDate:[000001010000 TO {to}]")),
        (None, None) => {}
    }

    if clauses.is_empty() {
        return Err(AppError::domain(
            "usage",
            "discover: provide at least one keyword or category",
        ));
    }
    Ok(clauses.join(" AND "))
}

/// `YYYY-MM-DD` → arXiv `submittedDate` bound (`YYYYMMDD0000` / `YYYYMMDD2359`).
fn date_bound(date: &str, end: bool) -> Result<String, AppError> {
    let parsed = chrono::NaiveDate::parse_from_str(date.trim(), "%Y-%m-%d").map_err(|_| {
        AppError::domain(
            "usage",
            format!("discover: invalid date '{date}' (expected YYYY-MM-DD)"),
        )
    })?;
    let suffix = if end { "2359" } else { "0000" };
    Ok(format!("{}{suffix}", parsed.format("%Y%m%d")))
}

/// Lowercased, de-duplicated, non-empty query terms.
fn normalize_terms(keywords: &[String]) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    for keyword in keywords {
        let term = keyword.trim().to_lowercase().replace('"', "");
        if term.is_empty() || out.iter().any(|existing| existing == &term) {
            continue;
        }
        out.push(term);
    }
    out
}

fn count_occurrences(haystack: &str, needle: &str) -> u32 {
    if needle.is_empty() {
        return 0;
    }
    haystack.matches(needle).count() as u32
}

/// Score candidates against the normalized `terms` (no sort / truncate).
/// With no terms every score is zero, so [`finalize`] preserves the
/// submission-date ordering from the arXiv query.
fn score_lexical(papers: Vec<ApiPaper>, terms: &[String]) -> Vec<DiscoverItem> {
    papers
        .into_iter()
        .map(|paper| {
            let title = paper.title.trim().to_string();
            let abstract_text = paper.abstract_text.unwrap_or_default().trim().to_string();
            let title_lower = title.to_lowercase();
            let abstract_lower = abstract_text.to_lowercase();

            let mut matches: Vec<ScoreTerm> = Vec::new();
            let mut score = 0.0_f32;
            for term in terms {
                let title_hits = count_occurrences(&title_lower, term);
                let abstract_hits = count_occurrences(&abstract_lower, term);
                if title_hits == 0 && abstract_hits == 0 {
                    continue;
                }
                if title_hits > 0 {
                    let contribution = title_hits as f32 * TITLE_WEIGHT;
                    score += contribution;
                    matches.push(ScoreTerm {
                        term: term.clone(),
                        field: "title".to_string(),
                        count: title_hits,
                        weight: TITLE_WEIGHT,
                        contribution,
                    });
                }
                if abstract_hits > 0 {
                    let contribution = abstract_hits as f32 * ABSTRACT_WEIGHT;
                    score += contribution;
                    matches.push(ScoreTerm {
                        term: term.clone(),
                        field: "abstract".to_string(),
                        count: abstract_hits,
                        weight: ABSTRACT_WEIGHT,
                        contribution,
                    });
                }
            }

            DiscoverItem {
                arxiv_id: paper.identifiers.arxiv_id.unwrap_or_default(),
                doi: paper.identifiers.doi.clone(),
                title,
                abstract_text,
                url: paper.urls.landing.unwrap_or_default(),
                pdf_url: paper.urls.pdf,
                published_at: paper.date,
                score,
                matches,
            }
        })
        .collect()
}

/// Sort by score (desc), then newest, then title; truncate to `top`.
fn finalize(mut items: Vec<DiscoverItem>, top: usize) -> Vec<DiscoverItem> {
    items.sort_by(|a, b| {
        b.score
            .total_cmp(&a.score)
            .then_with(|| b.published_at.cmp(&a.published_at))
            .then_with(|| a.title.cmp(&b.title))
    });
    items.truncate(top);
    items
}

/// Lexical-only ranking helper (unit tests).
#[cfg(test)]
fn rank(papers: Vec<ApiPaper>, keywords: &[String], top: usize) -> Vec<DiscoverItem> {
    finalize(score_lexical(papers, &normalize_terms(keywords)), top)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::features::scholar_api::{PaperIdentifiers, PaperUrls};

    fn paper(id: &str, title: &str, abstract_text: &str, date: &str) -> ApiPaper {
        ApiPaper {
            identifiers: PaperIdentifiers {
                arxiv_id: Some(id.to_string()),
                ..Default::default()
            },
            title: title.to_string(),
            abstract_text: Some(abstract_text.to_string()),
            date: Some(date.to_string()),
            urls: PaperUrls {
                landing: Some(format!("https://arxiv.org/abs/{id}")),
                pdf: Some(format!("https://arxiv.org/pdf/{id}")),
                ..Default::default()
            },
            source: "arxiv",
            ..Default::default()
        }
    }

    #[test]
    fn builds_keyword_category_and_date_clauses() {
        let query = DiscoverQuery {
            keywords: vec!["agent".into(), "world model".into()],
            categories: vec!["cs.AI".into()],
            since: Some("2026-08-01".into()),
            until: Some("2026-08-07".into()),
            ..Default::default()
        };
        assert_eq!(
            build_search_query(&query).unwrap(),
            "all:agent AND all:\"world model\" AND cat:cs.AI \
             AND submittedDate:[202608010000 TO 202608072359]"
        );
    }

    #[test]
    fn date_bound_rejects_bad_dates() {
        let err = build_search_query(&DiscoverQuery {
            keywords: vec!["agent".into()],
            since: Some("2026/08/01".into()),
            ..Default::default()
        })
        .unwrap_err();
        assert_eq!(err.code(), "usage");
    }

    #[test]
    fn empty_query_is_usage_error() {
        let err = build_search_query(&DiscoverQuery::default()).unwrap_err();
        assert_eq!(err.code(), "usage");
    }

    #[test]
    fn title_hits_outweigh_abstract_hits() {
        let candidates = vec![
            paper("1", "agentic search", "", "2026-08-01T00:00:00Z"),
            paper(
                "2",
                "unrelated",
                "we study agent behaviour",
                "2026-08-02T00:00:00Z",
            ),
        ];
        let ranked = rank(candidates, &["agent".to_string()], 8);
        assert_eq!(ranked[0].arxiv_id, "1");
        assert!(ranked[0].score > ranked[1].score);
        // Title match carries the 3x weight.
        assert_eq!(ranked[0].matches[0].field, "title");
        assert_eq!(ranked[0].matches[0].weight, TITLE_WEIGHT);
    }

    #[test]
    fn no_keywords_preserve_newest_first() {
        let candidates = vec![
            paper("old", "a", "", "2026-08-01T00:00:00Z"),
            paper("new", "b", "", "2026-08-05T00:00:00Z"),
        ];
        let ranked = rank(candidates, &[], 8);
        assert_eq!(ranked[0].arxiv_id, "new");
    }

    #[test]
    fn rank_truncates_to_top() {
        let candidates = (0..10)
            .map(|i| paper(&i.to_string(), "agent", "", "2026-08-01T00:00:00Z"))
            .collect();
        assert_eq!(rank(candidates, &["agent".to_string()], 3).len(), 3);
    }

    #[test]
    fn phrase_term_matches_contiguous_text() {
        let candidates = vec![paper(
            "1",
            "a world model for agents",
            "",
            "2026-08-01T00:00:00Z",
        )];
        let ranked = rank(candidates, &["world model".to_string()], 8);
        assert!(!ranked[0].matches.is_empty());
    }

    #[test]
    fn semantic_vectors_add_cosine_and_reorder() {
        // Two candidates with equal lexical score; semantic decides.
        let candidates = vec![
            paper("low", "agent", "", "2026-08-01T00:00:00Z"),
            paper("high", "agent", "", "2026-08-01T00:00:00Z"),
        ];
        let mut items = score_lexical(candidates, &["agent".to_string()]);
        let query = vec![1.0_f32, 0.0];
        let vectors = vec![vec![0.1_f32, 0.0], vec![0.9_f32, 0.0]];
        apply_semantic_vectors(&mut items, &query, &vectors, 2.0);

        let ranked = finalize(items, 8);
        assert_eq!(ranked[0].arxiv_id, "high");
        assert!((ranked[0].score - (3.0 + 1.8)).abs() < 1e-5);
        assert_eq!(ranked[0].matches.last().unwrap().field, "semantic");
    }

    #[test]
    fn semantic_dimension_mismatch_is_zero() {
        let candidates = vec![paper("a", "agent", "", "2026-08-01T00:00:00Z")];
        let mut items = score_lexical(candidates, &["agent".to_string()]);
        let before = items[0].score;
        apply_semantic_vectors(&mut items, &[1.0, 0.0], &[vec![0.5]], 1.0);
        assert!((items[0].score - before).abs() < 1e-6);
    }

    #[test]
    fn drop_known_removes_library_ids_only() {
        let candidates = vec![
            paper("known", "a", "", "2026-08-01T00:00:00Z"),
            paper("fresh", "b", "", "2026-08-01T00:00:00Z"),
        ];
        let exclude = Exclusion {
            known: ["arxiv:known".to_string()].into_iter().collect(),
            ..Default::default()
        };
        let (kept, dropped, seen_dropped) = drop_known(candidates, &exclude);
        assert_eq!((dropped, seen_dropped), (1, 0));
        assert_eq!(kept.len(), 1);
        assert_eq!(kept[0].identifiers.arxiv_id.as_deref(), Some("fresh"));

        // Empty exclusion is a no-op.
        let single = vec![paper("x", "t", "", "2026-08-01T00:00:00Z")];
        let (kept, dropped, seen_dropped) = drop_known(single, &Exclusion::default());
        assert_eq!((kept.len(), dropped, seen_dropped), (1, 0, 0));
    }

    #[test]
    fn drop_known_matches_ledger_by_arxiv_id_version_insensitive() {
        // An entry recorded with a version-suffixed id is stored normalized,
        // so a candidate carrying the bare id still matches.
        let mut ledger = crate::features::paper::discovery::seen::SeenLedger::new();
        ledger.add([crate::features::paper::discovery::seen::SeenEntry {
            title: "recorded under a different phrasing".to_string(),
            arxiv_id: Some("arXiv:2401.12345v2".to_string()),
            doi: None,
            source: Some("arxiv".to_string()),
            at: "2026-08-01T00:00:00.000Z".to_string(),
        }]);
        let candidates = vec![paper("2401.12345", "t", "", "2026-08-01T00:00:00Z")];
        let exclude = Exclusion {
            seen: ledger.keys().clone(),
            ..Default::default()
        };
        let (kept, dropped, seen_dropped) = drop_known(candidates, &exclude);
        assert_eq!((kept.len(), dropped, seen_dropped), (0, 0, 1));
    }

    #[test]
    fn drop_known_is_cross_source_for_journal_ledger_entries() {
        // A journal RSS item has no arXiv id: its ledger entry matches by
        // DOI and by normalized title. Both must drop the arXiv preprint.
        let mut preprint = paper(
            "2401.12345",
            "Dance and Choreography: A Review",
            "",
            "2026-08-01T00:00:00Z",
        );
        preprint.identifiers.doi = Some("https://doi.org/10.1000/dance.2026.1".to_string());

        // DOI-only ledger entry (feed item whose title the radar normalized
        // differently is still caught by the DOI key).
        let by_doi = Exclusion {
            seen: ["doi:10.1000/dance.2026.1".to_string()]
                .into_iter()
                .collect(),
            ..Default::default()
        };
        let (kept, _, seen_dropped) = drop_known(vec![preprint.clone()], &by_doi);
        assert_eq!((kept.len(), seen_dropped), (0, 1));

        // Title-only ledger entry: journal title differs only in case and
        // punctuation from the arXiv title.
        let by_title = Exclusion {
            seen: [
                identity::tagged_keys(None, None, "dance and choreography — a review")[0].clone(),
            ]
            .into_iter()
            .collect(),
            ..Default::default()
        };
        let (kept, _, seen_dropped) = drop_known(vec![preprint], &by_title);
        assert_eq!((kept.len(), seen_dropped), (0, 1));
    }

    #[test]
    fn drop_known_attributes_library_filter_first() {
        let candidates = vec![paper("2401.12345", "t", "", "2026-08-01T00:00:00Z")];
        let exclude = Exclusion {
            known: ["arxiv:2401.12345".to_string()].into_iter().collect(),
            seen: ["arxiv:2401.12345".to_string()].into_iter().collect(),
        };
        let (kept, dropped, seen_dropped) = drop_known(candidates, &exclude);
        assert_eq!((kept.len(), dropped, seen_dropped), (0, 1, 0));
    }
}
