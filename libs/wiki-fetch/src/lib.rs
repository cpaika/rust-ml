//! Wikipedia Fetcher
//!
//! Fetches and cleans Wikipedia articles for language model training.
//! Uses the Wikipedia API to retrieve article content and strips markup.

use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::fs;
use std::path::Path;

mod cleaner;

pub use cleaner::WikiCleaner;

/// Configuration for Wikipedia fetching
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FetchConfig {
    /// Target size in bytes (default: 100MB)
    pub target_size: usize,
    /// Minimum article length to include (in characters)
    pub min_article_length: usize,
    /// Maximum articles to fetch (safety limit)
    pub max_articles: usize,
    /// Language code (e.g., "en" for English)
    pub language: String,
    /// User agent for API requests
    pub user_agent: String,
}

impl Default for FetchConfig {
    fn default() -> Self {
        Self {
            target_size: 100 * 1024 * 1024, // 100 MB
            min_article_length: 1000,
            max_articles: 50000,
            language: "en".to_string(),
            user_agent: "RustMLTrainer/1.0 (https://github.com/rust-ml; contact@example.com)".to_string(),
        }
    }
}

/// Error type for wiki-fetch operations
#[derive(Debug)]
pub enum WikiError {
    /// HTTP request failed
    HttpError(String),
    /// JSON parsing failed
    ParseError(String),
    /// IO error
    IoError(std::io::Error),
    /// No more articles available
    NoMoreArticles,
}

impl std::fmt::Display for WikiError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            WikiError::HttpError(e) => write!(f, "HTTP error: {}", e),
            WikiError::ParseError(e) => write!(f, "Parse error: {}", e),
            WikiError::IoError(e) => write!(f, "IO error: {}", e),
            WikiError::NoMoreArticles => write!(f, "No more articles available"),
        }
    }
}

impl std::error::Error for WikiError {}

impl From<std::io::Error> for WikiError {
    fn from(e: std::io::Error) -> Self {
        WikiError::IoError(e)
    }
}

/// Response from Wikipedia random articles API
#[derive(Debug, Deserialize)]
struct RandomResponse {
    query: Option<RandomQuery>,
}

#[derive(Debug, Deserialize)]
struct RandomQuery {
    random: Option<Vec<RandomArticle>>,
}

#[derive(Debug, Deserialize)]
struct RandomArticle {
    id: u64,
    #[allow(dead_code)]
    title: String,
}

/// Response from Wikipedia parse API
#[derive(Debug, Deserialize)]
struct ParseResponse {
    parse: Option<ParseResult>,
}

#[derive(Debug, Deserialize)]
struct ParseResult {
    title: String,
    text: Option<ParseText>,
}

#[derive(Debug, Deserialize)]
struct ParseText {
    #[serde(rename = "*")]
    content: String,
}

/// A fetched and cleaned Wikipedia article
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Article {
    /// Article title
    pub title: String,
    /// Wikipedia page ID
    pub page_id: u64,
    /// Cleaned article text
    pub text: String,
}

impl Article {
    /// Get the byte length of the article text
    pub fn byte_len(&self) -> usize {
        self.text.len()
    }
}

/// Wikipedia article fetcher
pub struct WikiFetcher {
    config: FetchConfig,
    cleaner: WikiCleaner,
    seen_ids: HashSet<u64>,
}

impl WikiFetcher {
    /// Create a new fetcher with the given configuration
    pub fn new(config: FetchConfig) -> Self {
        Self {
            config,
            cleaner: WikiCleaner::new(),
            seen_ids: HashSet::new(),
        }
    }

    /// Create a fetcher with default configuration
    pub fn with_defaults() -> Self {
        Self::new(FetchConfig::default())
    }

    /// Get the API base URL for the configured language
    fn api_url(&self) -> String {
        format!("https://{}.wikipedia.org/w/api.php", self.config.language)
    }

    /// Fetch random article titles from Wikipedia
    fn fetch_random_titles(&self, count: usize) -> Result<Vec<RandomArticle>, WikiError> {
        let url = format!(
            "{}?action=query&list=random&rnnamespace=0&rnlimit={}&format=json",
            self.api_url(),
            count
        );

        let response: RandomResponse = ureq::get(&url)
            .header("User-Agent", &self.config.user_agent)
            .call()
            .map_err(|e| WikiError::HttpError(e.to_string()))?
            .body_mut()
            .read_json()
            .map_err(|e| WikiError::ParseError(e.to_string()))?;

        response
            .query
            .and_then(|q| q.random)
            .ok_or(WikiError::NoMoreArticles)
    }

    /// Fetch and parse a single article by page ID
    fn fetch_article(&self, page_id: u64) -> Result<Option<Article>, WikiError> {
        let url = format!(
            "{}?action=parse&pageid={}&prop=text&format=json",
            self.api_url(),
            page_id
        );

        let response: ParseResponse = ureq::get(&url)
            .header("User-Agent", &self.config.user_agent)
            .call()
            .map_err(|e| WikiError::HttpError(e.to_string()))?
            .body_mut()
            .read_json()
            .map_err(|e| WikiError::ParseError(e.to_string()))?;

        let Some(parse) = response.parse else {
            return Ok(None);
        };

        let Some(text) = parse.text else {
            return Ok(None);
        };

        // Clean the HTML content
        let cleaned = self.cleaner.clean(&text.content);

        // Skip short articles
        if cleaned.len() < self.config.min_article_length {
            return Ok(None);
        }

        Ok(Some(Article {
            title: parse.title,
            page_id,
            text: cleaned,
        }))
    }

    /// Fetch articles until target size is reached
    pub fn fetch_corpus(&mut self) -> Result<Vec<Article>, WikiError> {
        let mut articles = Vec::new();
        let mut total_size = 0usize;
        let batch_size = 50;

        println!("Starting Wikipedia corpus fetch...");
        println!("Target size: {} MB", self.config.target_size / 1024 / 1024);

        while total_size < self.config.target_size && articles.len() < self.config.max_articles {
            // Fetch a batch of random article titles
            let random_articles = self.fetch_random_titles(batch_size)?;

            for random in random_articles {
                if self.seen_ids.contains(&random.id) {
                    continue;
                }
                self.seen_ids.insert(random.id);

                match self.fetch_article(random.id) {
                    Ok(Some(article)) => {
                        total_size += article.byte_len();
                        println!(
                            "Fetched: {} ({} KB) - Total: {} MB",
                            article.title,
                            article.byte_len() / 1024,
                            total_size / 1024 / 1024
                        );
                        articles.push(article);

                        if total_size >= self.config.target_size {
                            break;
                        }
                    }
                    Ok(None) => continue,
                    Err(e) => {
                        eprintln!("Error fetching article {}: {}", random.id, e);
                        continue;
                    }
                }

                // Be nice to Wikipedia's servers
                std::thread::sleep(std::time::Duration::from_millis(100));
            }
        }

        println!(
            "Corpus fetch complete: {} articles, {} MB",
            articles.len(),
            total_size / 1024 / 1024
        );

        Ok(articles)
    }

    /// Save corpus to a file
    pub fn save_corpus(articles: &[Article], path: &Path) -> Result<(), WikiError> {
        let json = serde_json::to_string_pretty(articles)
            .map_err(|e| WikiError::ParseError(e.to_string()))?;
        fs::write(path, json)?;
        Ok(())
    }

    /// Load corpus from a file
    pub fn load_corpus(path: &Path) -> Result<Vec<Article>, WikiError> {
        let json = fs::read_to_string(path)?;
        serde_json::from_str(&json).map_err(|e| WikiError::ParseError(e.to_string()))
    }

    /// Save corpus as plain text (for embedding in WASM)
    pub fn save_corpus_text(articles: &[Article], path: &Path) -> Result<(), WikiError> {
        let mut text = String::new();
        for article in articles {
            text.push_str(&article.title);
            text.push_str("\n\n");
            text.push_str(&article.text);
            text.push_str("\n\n---\n\n");
        }
        fs::write(path, text)?;
        Ok(())
    }

    /// Load plain text corpus
    pub fn load_corpus_text(path: &Path) -> Result<String, WikiError> {
        Ok(fs::read_to_string(path)?)
    }
}

/// Corpus for training with iterator support
#[derive(Debug, Clone)]
pub struct Corpus {
    /// Raw text content
    pub text: String,
}

impl Corpus {
    /// Create a corpus from articles
    pub fn from_articles(articles: &[Article]) -> Self {
        let mut text = String::new();
        for article in articles {
            text.push_str(&article.text);
            text.push_str("\n\n");
        }
        Self { text }
    }

    /// Create a corpus from raw text
    pub fn from_text(text: String) -> Self {
        Self { text }
    }

    /// Load corpus from file
    pub fn load(path: &Path) -> Result<Self, WikiError> {
        let text = fs::read_to_string(path)?;
        Ok(Self { text })
    }

    /// Save corpus to file
    pub fn save(&self, path: &Path) -> Result<(), WikiError> {
        fs::write(path, &self.text)?;
        Ok(())
    }

    /// Get the total size in bytes
    pub fn size(&self) -> usize {
        self.text.len()
    }

    /// Split into chunks for training
    pub fn chunks(&self, chunk_size: usize) -> Vec<&str> {
        let chars: Vec<char> = self.text.chars().collect();
        let mut chunks = Vec::new();
        let mut start = 0;

        while start < chars.len() {
            let end = std::cmp::min(start + chunk_size, chars.len());
            // Convert back to string slice using byte indices
            let start_byte = chars[..start].iter().map(|c| c.len_utf8()).sum::<usize>();
            let end_byte = start_byte + chars[start..end].iter().map(|c| c.len_utf8()).sum::<usize>();
            chunks.push(&self.text[start_byte..end_byte]);
            start = end;
        }

        chunks
    }

    /// Get text references for BPE training
    pub fn as_training_texts(&self, max_chunk_size: usize) -> Vec<&str> {
        if self.text.len() <= max_chunk_size {
            vec![&self.text]
        } else {
            self.chunks(max_chunk_size)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fetch_config_default() {
        let config = FetchConfig::default();
        assert_eq!(config.target_size, 100 * 1024 * 1024);
        assert_eq!(config.min_article_length, 1000);
        assert_eq!(config.language, "en");
    }

    #[test]
    fn test_article_byte_len() {
        let article = Article {
            title: "Test".to_string(),
            page_id: 1,
            text: "Hello, world!".to_string(),
        };
        assert_eq!(article.byte_len(), 13);
    }

    #[test]
    fn test_corpus_from_articles() {
        let articles = vec![
            Article {
                title: "A".to_string(),
                page_id: 1,
                text: "Article A content".to_string(),
            },
            Article {
                title: "B".to_string(),
                page_id: 2,
                text: "Article B content".to_string(),
            },
        ];

        let corpus = Corpus::from_articles(&articles);
        assert!(corpus.text.contains("Article A content"));
        assert!(corpus.text.contains("Article B content"));
    }

    #[test]
    fn test_corpus_chunks() {
        let corpus = Corpus::from_text("Hello world! This is a test.".to_string());
        let chunks = corpus.chunks(10);
        assert!(!chunks.is_empty());
        // All chunks should be <= 10 chars
        for chunk in &chunks {
            assert!(chunk.chars().count() <= 10);
        }
    }

    #[test]
    fn test_corpus_save_load() {
        let corpus = Corpus::from_text("Test corpus content".to_string());
        let temp_dir = tempfile::tempdir().unwrap();
        let path = temp_dir.path().join("corpus.txt");

        corpus.save(&path).unwrap();
        let loaded = Corpus::load(&path).unwrap();

        assert_eq!(loaded.text, corpus.text);
    }

    #[test]
    fn test_wiki_error_display() {
        let err = WikiError::HttpError("connection refused".to_string());
        assert!(format!("{}", err).contains("connection refused"));

        let err = WikiError::NoMoreArticles;
        assert!(format!("{}", err).contains("No more articles"));
    }

    #[test]
    fn test_corpus_size() {
        let corpus = Corpus::from_text("Hello".to_string());
        assert_eq!(corpus.size(), 5);
    }

    #[test]
    fn test_corpus_unicode() {
        let corpus = Corpus::from_text("Hello 世界".to_string());
        let chunks = corpus.chunks(5);
        // Should handle unicode properly
        assert!(!chunks.is_empty());
    }

    #[test]
    fn test_as_training_texts_small() {
        let corpus = Corpus::from_text("Small text".to_string());
        let texts = corpus.as_training_texts(1000);
        assert_eq!(texts.len(), 1);
        assert_eq!(texts[0], "Small text");
    }

    #[test]
    fn test_as_training_texts_large() {
        let corpus = Corpus::from_text("A".repeat(100));
        let texts = corpus.as_training_texts(10);
        assert_eq!(texts.len(), 10);
    }
}
