//! Normalized paper-identity keys for cross-source deduplication.
//!
//! The same paper arrives through different sources wearing different
//! identifiers: an arXiv submission carries an arXiv id, a journal RSS item
//! carries a DOI, and the title is the only field both always have. Exact
//! arXiv-id comparison therefore silently misses every arXiv↔journal pair.
//!
//! Matching is key-based instead: each paper yields a small set of tagged,
//! normalized keys (`arxiv:…`, `doi:…`, `title:…`) and two papers are the
//! same when any of their keys agree.

/// `arXiv:2401.12345v2` / `ARXIV:2401.12345` / `2401.12345` → `2401.12345`.
///
/// Strips the optional `arXiv:` prefix and a trailing version suffix. Old
/// style ids (`hep-th/9901001`) pass through unchanged.
pub fn normalize_arxiv_id(raw: &str) -> Option<String> {
    let mut id = raw.trim().to_ascii_lowercase();
    if let Some(stripped) = id.strip_prefix("arxiv:") {
        id = stripped.trim().to_ascii_lowercase();
    }
    if let Some(pos) = id.rfind('v') {
        let suffix = &id[pos + 1..];
        if !suffix.is_empty() && suffix.chars().all(|c| c.is_ascii_digit()) {
            id = id[..pos].to_string();
        }
    }
    let id = id.trim();
    if id.is_empty() {
        None
    } else {
        Some(id.to_string())
    }
}

/// `https://doi.org/10.1000/XYZ` / `doi:10.1000/xyz` / `10.1000/XYZ`
/// → `10.1000/xyz`.
///
/// Lowercases and strips resolver prefixes; rejects anything that does not
/// start with the `10.` registration prefix so bare URLs, GUIDs and file
/// paths never become dedup keys.
pub fn normalize_doi(raw: &str) -> Option<String> {
    let mut doi = raw.trim().to_ascii_lowercase();
    for prefix in [
        "https://doi.org/",
        "http://doi.org/",
        "https://dx.doi.org/",
        "http://dx.doi.org/",
        "doi.org/",
        "doi:",
    ] {
        if let Some(stripped) = doi.strip_prefix(prefix) {
            doi = stripped.trim().to_string();
            break;
        }
    }
    doi = doi.trim_matches('/').trim().to_string();
    if doi.starts_with("10.") && doi.len() > 3 {
        Some(doi)
    } else {
        None
    }
}

/// Case/punctuation-insensitive title key: lowercase, keep word characters,
/// collapse whitespace runs. Punctuation acts as a word boundary (not glue),
/// so "chain-of-dancers", "chain of dancers" and "chain  of  dancers" all
/// key identically. `Some` only when at least one word character survives,
/// so empty or all-punctuation titles yield no key.
pub fn normalize_title(raw: &str) -> Option<String> {
    let mut out = String::with_capacity(raw.len());
    let mut in_word = false;
    for c in raw.chars() {
        if c.is_alphanumeric() {
            if in_word {
                out.extend(c.to_lowercase());
            } else {
                // First char of a new word: separate from whatever came
                // before (word, space or punctuation).
                if !out.is_empty() {
                    out.push(' ');
                }
                out.extend(c.to_lowercase());
                in_word = true;
            }
        } else {
            // Whitespace and punctuation alike act as separators, so
            // "chain-of-dancers", "chain  of  dancers" and
            // "Choreography: A" all key identically.
            in_word = false;
        }
    }
    let out = out.trim().to_string();
    if out.is_empty() {
        None
    } else {
        Some(out)
    }
}

/// Tagged identity keys for one paper, in `arxiv`, `doi`, `title` order.
pub fn tagged_keys(arxiv_id: Option<&str>, doi: Option<&str>, title: &str) -> Vec<String> {
    let mut keys = Vec::with_capacity(3);
    if let Some(id) = arxiv_id.and_then(normalize_arxiv_id) {
        keys.push(format!("arxiv:{id}"));
    }
    if let Some(d) = doi.and_then(normalize_doi) {
        keys.push(format!("doi:{d}"));
    }
    if let Some(t) = normalize_title(title) {
        keys.push(format!("title:{t}"));
    }
    keys
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn arxiv_id_strips_prefix_and_version() {
        assert_eq!(
            normalize_arxiv_id("arXiv:2401.12345v2"),
            Some("2401.12345".into())
        );
        assert_eq!(
            normalize_arxiv_id("ARXIV:2401.12345"),
            Some("2401.12345".into())
        );
        assert_eq!(
            normalize_arxiv_id("2401.12345v10"),
            Some("2401.12345".into())
        );
        // Old style id without version passes through.
        assert_eq!(
            normalize_arxiv_id("hep-th/9901001"),
            Some("hep-th/9901001".into())
        );
        assert_eq!(normalize_arxiv_id("  "), None);
        assert_eq!(normalize_arxiv_id("arXiv:"), None);
    }

    #[test]
    fn doi_strips_resolver_prefixes_and_lowercases() {
        assert_eq!(
            normalize_doi("https://doi.org/10.1000/XYZ"),
            Some("10.1000/xyz".into())
        );
        assert_eq!(
            normalize_doi("http://dx.doi.org/10.1038/s41467-026-76837-1"),
            Some("10.1038/s41467-026-76837-1".into())
        );
        assert_eq!(normalize_doi("doi:10.1/abc"), Some("10.1/abc".into()));
        // Not a DOI: no 10. prefix.
        assert_eq!(normalize_doi("https://example.com/abs/1"), None);
        assert_eq!(normalize_doi("0a1b2c3d-uuid"), None);
        assert_eq!(normalize_doi(""), None);
    }

    #[test]
    fn title_ignores_case_punctuation_and_spacing() {
        let a = normalize_title("Dance and Choreography: A Review of 20 Years!").unwrap();
        let b = normalize_title("dance  and choreography — a review of 20 years").unwrap();
        assert_eq!(a, b);
        assert_eq!(a, "dance and choreography a review of 20 years");
        assert_eq!(normalize_title("   "), None);
        assert_eq!(normalize_title("???"), None);
    }

    #[test]
    fn title_punctuation_is_a_word_boundary() {
        // Hyphen, em-dash and extra spacing must key identically.
        assert_eq!(
            normalize_title("Chain-of-Dancers"),
            Some("chain of dancers".into())
        );
        assert_eq!(
            normalize_title("chain  of  dancers"),
            Some("chain of dancers".into())
        );
        assert_eq!(
            normalize_title("chain — of — dancers"),
            Some("chain of dancers".into())
        );
    }

    #[test]
    fn tagged_keys_cover_every_identifier_a_paper_has() {
        let keys = tagged_keys(
            Some("arXiv:2401.12345v2"),
            Some("https://doi.org/10.1/x"),
            "T",
        );
        assert_eq!(keys, vec!["arxiv:2401.12345", "doi:10.1/x", "title:t"]);
        let feed_only = tagged_keys(None, None, "Some Paper Title");
        assert_eq!(feed_only, vec!["title:some paper title"]);
    }
}
