//! HTML and Wiki markup cleaner

use regex::Regex;

/// Cleans Wikipedia HTML content into plain text
pub struct WikiCleaner {
    // Regex patterns for cleaning
    html_tags: Regex,
    html_comments: Regex,
    wiki_templates: Regex,
    wiki_refs: Regex,
    wiki_links: Regex,
    wiki_files: Regex,
    wiki_categories: Regex,
    wiki_tables: Regex,
    multiple_newlines: Regex,
    multiple_spaces: Regex,
    #[allow(dead_code)]
    html_entities: Regex,
}

impl WikiCleaner {
    /// Create a new wiki cleaner
    pub fn new() -> Self {
        Self {
            // Remove HTML tags
            html_tags: Regex::new(r"<[^>]+>").unwrap(),
            // Remove HTML comments
            html_comments: Regex::new(r"<!--[\s\S]*?-->").unwrap(),
            // Remove wiki templates {{...}}
            wiki_templates: Regex::new(r"\{\{[^{}]*\}\}").unwrap(),
            // Remove references [1], [citation needed], etc.
            wiki_refs: Regex::new(r"\[[^\]]*\]").unwrap(),
            // Convert wiki links [[link|text]] -> text
            wiki_links: Regex::new(r"\[\[(?:[^|\]]*\|)?([^\]]+)\]\]").unwrap(),
            // Remove file/image references
            wiki_files: Regex::new(r"\[\[(?:File|Image|Media):[^\]]+\]\]").unwrap(),
            // Remove category links
            wiki_categories: Regex::new(r"\[\[Category:[^\]]+\]\]").unwrap(),
            // Remove wiki tables
            wiki_tables: Regex::new(r"\{\|[\s\S]*?\|\}").unwrap(),
            // Collapse multiple newlines
            multiple_newlines: Regex::new(r"\n{3,}").unwrap(),
            // Collapse multiple spaces
            multiple_spaces: Regex::new(r" {2,}").unwrap(),
            // HTML entities
            html_entities: Regex::new(r"&[a-zA-Z]+;|&#[0-9]+;").unwrap(),
        }
    }

    /// Clean HTML content into plain text
    pub fn clean(&self, html: &str) -> String {
        let mut text = html.to_string();

        // Remove HTML comments first
        text = self.html_comments.replace_all(&text, "").to_string();

        // Remove wiki tables
        text = self.wiki_tables.replace_all(&text, "").to_string();

        // Remove file/image references
        text = self.wiki_files.replace_all(&text, "").to_string();

        // Remove category links
        text = self.wiki_categories.replace_all(&text, "").to_string();

        // Remove templates (may be nested, so run multiple times)
        for _ in 0..5 {
            let new_text = self.wiki_templates.replace_all(&text, "").to_string();
            if new_text == text {
                break;
            }
            text = new_text;
        }

        // Convert wiki links to just the text
        text = self.wiki_links.replace_all(&text, "$1").to_string();

        // Remove HTML tags
        text = self.html_tags.replace_all(&text, "").to_string();

        // Decode common HTML entities
        text = self.decode_entities(&text);

        // Remove references
        text = self.wiki_refs.replace_all(&text, "").to_string();

        // Collapse whitespace
        text = self.multiple_newlines.replace_all(&text, "\n\n").to_string();
        text = self.multiple_spaces.replace_all(&text, " ").to_string();

        // Trim each line
        text = text
            .lines()
            .map(|line| line.trim())
            .collect::<Vec<_>>()
            .join("\n");

        // Final trim
        text.trim().to_string()
    }

    /// Decode common HTML entities
    fn decode_entities(&self, text: &str) -> String {
        text.replace("&nbsp;", " ")
            .replace("&amp;", "&")
            .replace("&lt;", "<")
            .replace("&gt;", ">")
            .replace("&quot;", "\"")
            .replace("&apos;", "'")
            .replace("&#39;", "'")
            .replace("&#34;", "\"")
            .replace("&ndash;", "-")
            .replace("&mdash;", "-")
            .replace("&hellip;", "...")
            .replace("&copy;", "(c)")
            .replace("&reg;", "(R)")
            .replace("&trade;", "(TM)")
    }

    /// Extract just the first paragraph (for summaries)
    pub fn extract_summary(&self, html: &str) -> String {
        let cleaned = self.clean(html);
        cleaned
            .split("\n\n")
            .find(|p| p.len() > 100)
            .unwrap_or("")
            .to_string()
    }
}

impl Default for WikiCleaner {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_remove_html_tags() {
        let cleaner = WikiCleaner::new();
        let html = "<p>Hello <b>world</b>!</p>";
        let cleaned = cleaner.clean(html);
        assert_eq!(cleaned, "Hello world!");
    }

    #[test]
    fn test_remove_html_comments() {
        let cleaner = WikiCleaner::new();
        let html = "Before <!-- comment --> After";
        let cleaned = cleaner.clean(html);
        assert_eq!(cleaned, "Before After");
    }

    #[test]
    fn test_wiki_link_conversion() {
        let cleaner = WikiCleaner::new();

        // Simple link
        let text = "Visit [[Wikipedia]] for more info.";
        let cleaned = cleaner.clean(text);
        assert!(cleaned.contains("Wikipedia"));

        // Link with display text
        let text = "Visit [[Wikipedia|the free encyclopedia]] for more info.";
        let cleaned = cleaner.clean(text);
        assert!(cleaned.contains("the free encyclopedia"));
        assert!(!cleaned.contains("[["));
    }

    #[test]
    fn test_remove_templates() {
        let cleaner = WikiCleaner::new();
        let text = "Hello {{citation needed}} world {{ref|date=2024}}";
        let cleaned = cleaner.clean(text);
        assert!(!cleaned.contains("{{"));
        assert!(!cleaned.contains("}}"));
        assert!(cleaned.contains("Hello"));
        assert!(cleaned.contains("world"));
    }

    #[test]
    fn test_remove_references() {
        let cleaner = WikiCleaner::new();
        let text = "This is a fact[1] with multiple sources[2][3].";
        let cleaned = cleaner.clean(text);
        assert!(!cleaned.contains("[1]"));
        assert!(!cleaned.contains("[2]"));
        assert!(cleaned.contains("This is a fact"));
    }

    #[test]
    fn test_remove_file_references() {
        let cleaner = WikiCleaner::new();
        let text = "Text [[File:Example.jpg|thumb|caption]] more text";
        let cleaned = cleaner.clean(text);
        assert!(!cleaned.contains("File:"));
        assert!(cleaned.contains("Text"));
        assert!(cleaned.contains("more text"));
    }

    #[test]
    fn test_remove_categories() {
        let cleaner = WikiCleaner::new();
        let text = "Article text. [[Category:Science]]";
        let cleaned = cleaner.clean(text);
        assert!(!cleaned.contains("Category:"));
        assert!(cleaned.contains("Article text"));
    }

    #[test]
    fn test_decode_entities() {
        let cleaner = WikiCleaner::new();
        let text = "Tom &amp; Jerry &ndash; a classic &copy; show";
        let cleaned = cleaner.clean(text);
        assert!(cleaned.contains("Tom & Jerry"));
        assert!(cleaned.contains("-"));
        assert!(cleaned.contains("(c)"));
    }

    #[test]
    fn test_collapse_whitespace() {
        let cleaner = WikiCleaner::new();
        let text = "Hello     world\n\n\n\n\nNew paragraph";
        let cleaned = cleaner.clean(text);
        assert!(!cleaned.contains("     "));
        assert!(cleaned.contains("Hello world"));
    }

    #[test]
    fn test_extract_summary() {
        let cleaner = WikiCleaner::new();
        // Use double newlines to create actual paragraph breaks
        let html = "<p>Short.</p>\n\n<p>This is the first real paragraph with enough content to be considered a summary. It contains more than one hundred characters of meaningful text for the reader.</p>\n\n<p>This is another paragraph with enough text to be long as well but should not appear in the summary at all.</p>";
        let summary = cleaner.extract_summary(html);
        assert!(summary.contains("first real paragraph"));
        // The summary should only be the first long paragraph
        assert!(!summary.contains("another paragraph"));
    }

    #[test]
    fn test_nested_templates() {
        let cleaner = WikiCleaner::new();
        let text = "Start {{outer {{inner}} template}} end";
        let cleaned = cleaner.clean(text);
        // Should remove templates even when nested
        assert!(!cleaned.contains("{{"));
    }

    #[test]
    fn test_complex_html() {
        let cleaner = WikiCleaner::new();
        let html = r#"
            <div class="article">
                <h1>Title</h1>
                <p>First paragraph with <a href="link">a link</a>.</p>
                <table><tr><td>Table content</td></tr></table>
                <p>Second paragraph.</p>
            </div>
        "#;
        let cleaned = cleaner.clean(html);
        assert!(cleaned.contains("Title"));
        assert!(cleaned.contains("First paragraph"));
        assert!(cleaned.contains("a link"));
        assert!(cleaned.contains("Second paragraph"));
    }

    #[test]
    fn test_empty_input() {
        let cleaner = WikiCleaner::new();
        let cleaned = cleaner.clean("");
        assert_eq!(cleaned, "");
    }

    #[test]
    fn test_plain_text_passthrough() {
        let cleaner = WikiCleaner::new();
        let text = "This is plain text without any markup.";
        let cleaned = cleaner.clean(text);
        assert_eq!(cleaned, text);
    }
}
