use serde::{Deserialize, Serialize};

/// Parsed SRT subtitle entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubtitleEntry {
    pub index: u32,
    pub start_ms: u64,
    pub end_ms: u64,
    pub text: String,
}

/// Parse SRT format subtitle content
pub fn parse_srt(content: &str) -> Vec<SubtitleEntry> {
    let mut entries = Vec::new();
    let blocks: Vec<&str> = content.split("\n\n").collect();

    for block in blocks {
        let lines: Vec<&str> = block.trim().lines().collect();
        if lines.len() < 3 {
            continue;
        }

        let index = match lines[0].trim().parse::<u32>() {
            Ok(i) => i,
            Err(_) => continue,
        };

        let timestamps: Vec<&str> = lines[1].split(" --> ").collect();
        if timestamps.len() != 2 {
            continue;
        }

        let start_ms = parse_timestamp(timestamps[0]);
        let end_ms = parse_timestamp(timestamps[1]);

        let text = lines[2..].join("\n");

        entries.push(SubtitleEntry {
            index,
            start_ms,
            end_ms,
            text,
        });
    }

    entries
}

fn parse_timestamp(ts: &str) -> u64 {
    let ts = ts.trim();
    let parts: Vec<&str> = ts.split(':').collect();
    if parts.len() != 3 {
        return 0;
    }

    let hours: u64 = parts[0].parse().unwrap_or(0);
    let minutes: u64 = parts[1].parse().unwrap_or(0);

    let sec_parts: Vec<&str> = parts[2].split(',').collect();
    let seconds: u64 = sec_parts[0].parse().unwrap_or(0);
    let millis: u64 = sec_parts.get(1).and_then(|s| s.parse().ok()).unwrap_or(0);

    hours * 3600000 + minutes * 60000 + seconds * 1000 + millis
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_srt_basic() {
        let srt = "1\n00:00:01,000 --> 00:00:04,000\nHello, world!\n\n2\n00:00:05,000 --> 00:00:08,000\nHow are you?";
        let entries = parse_srt(srt);
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].text, "Hello, world!");
        assert_eq!(entries[0].start_ms, 1000);
        assert_eq!(entries[0].end_ms, 4000);
        assert_eq!(entries[1].text, "How are you?");
        assert_eq!(entries[1].index, 2);
    }

    #[test]
    fn test_parse_srt_multiline_text() {
        let srt = "1\n00:00:01,000 --> 00:00:04,000\nLine one\nLine two";
        let entries = parse_srt(srt);
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].text, "Line one\nLine two");
    }

    #[test]
    fn test_parse_srt_empty() {
        let entries = parse_srt("");
        assert_eq!(entries.len(), 0);
    }

    #[test]
    fn test_parse_srt_malformed_skipped() {
        let srt = "not a number\n00:00:01,000 --> 00:00:04,000\nHello\n\n2\n00:00:05,000 --> 00:00:08,000\nWorld";
        let entries = parse_srt(srt);
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].text, "World");
    }

    #[test]
    fn test_parse_timestamp_hours() {
        assert_eq!(parse_timestamp("01:30:00,000"), 5400000);
    }

    #[test]
    fn test_parse_timestamp_millis() {
        assert_eq!(parse_timestamp("00:00:00,500"), 500);
    }

    #[test]
    fn test_parse_timestamp_full() {
        assert_eq!(parse_timestamp("02:15:33,250"), 8133250);
    }

    #[test]
    fn test_parse_timestamp_invalid() {
        assert_eq!(parse_timestamp("invalid"), 0);
    }

    #[test]
    fn test_subtitle_entry_serialization() {
        let entry = SubtitleEntry {
            index: 1,
            start_ms: 1000,
            end_ms: 4000,
            text: "Hello".to_string(),
        };
        let json = serde_json::to_string(&entry).unwrap();
        let deserialized: SubtitleEntry = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.index, 1);
        assert_eq!(deserialized.text, "Hello");
    }
}
