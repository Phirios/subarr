use serde::Deserialize;

/// Translated subtitle entry received from Python ML worker
#[derive(Debug, Deserialize)]
pub struct TranslatedEntry {
    pub start_ms: u64,
    pub end_ms: u64,
    pub text: String,
    pub speaker: Option<String>,
    pub emotion: Option<String>,
    pub color: Option<String>,  // hex color e.g. "FF0000"
}

/// Format translated entries to ASS/SSA subtitle format
pub fn format_ass(entries: &[TranslatedEntry], title: &str) -> String {
    let mut output = String::new();

    // Script Info
    output.push_str("[Script Info]\n");
    output.push_str(&format!("Title: {}\n", title));
    output.push_str("ScriptType: v4.00+\n");
    output.push_str("PlayResX: 1920\n");
    output.push_str("PlayResY: 1080\n");
    output.push_str("WrapStyle: 0\n\n");

    // Collect unique speakers for styles
    let mut speakers: Vec<String> = entries
        .iter()
        .filter_map(|e| e.speaker.clone())
        .collect::<std::collections::HashSet<_>>()
        .into_iter()
        .collect();
    speakers.sort();

    // Styles
    output.push_str("[V4+ Styles]\n");
    output.push_str("Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\n");
    output.push_str("Style: Default,Arial,60,&H00FFFFFF,&H000000FF,&H00000000,&H80000000,0,0,0,0,100,100,0,0,1,2,1,2,10,10,30,1\n");

    for entry in entries {
        if let (Some(speaker), Some(color)) = (&entry.speaker, &entry.color) {
            let ass_color = hex_to_ass_color(color);
            output.push_str(&format!(
                "Style: {},Arial,60,{},&H000000FF,&H00000000,&H80000000,0,0,0,0,100,100,0,0,1,2,1,2,10,10,30,1\n",
                speaker, ass_color
            ));
        }
    }

    output.push('\n');

    // Events
    output.push_str("[Events]\n");
    output.push_str("Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n");

    for entry in entries {
        let start = ms_to_ass_time(entry.start_ms);
        let end = ms_to_ass_time(entry.end_ms);
        let style = entry.speaker.as_deref().unwrap_or("Default");

        output.push_str(&format!(
            "Dialogue: 0,{},{},{},{},0,0,0,,{}\n",
            start, end, style, style, entry.text
        ));
    }

    output
}

fn ms_to_ass_time(ms: u64) -> String {
    let h = ms / 3600000;
    let m = (ms % 3600000) / 60000;
    let s = (ms % 60000) / 1000;
    let cs = (ms % 1000) / 10;
    format!("{}:{:02}:{:02}.{:02}", h, m, s, cs)
}

fn hex_to_ass_color(hex: &str) -> String {
    let hex = hex.trim_start_matches('#');
    if hex.len() != 6 {
        return "&H00FFFFFF".to_string();
    }
    // ASS color format: &HAABBGGRR (alpha, blue, green, red)
    let r = &hex[0..2];
    let g = &hex[2..4];
    let b = &hex[4..6];
    format!("&H00{}{}{}", b, g, r)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ms_to_ass_time() {
        assert_eq!(ms_to_ass_time(0), "0:00:00.00");
        assert_eq!(ms_to_ass_time(1000), "0:00:01.00");
        assert_eq!(ms_to_ass_time(3661500), "1:01:01.50");
        assert_eq!(ms_to_ass_time(60000), "0:01:00.00");
        assert_eq!(ms_to_ass_time(550), "0:00:00.55");
    }

    #[test]
    fn test_hex_to_ass_color() {
        assert_eq!(hex_to_ass_color("FF0000"), "&H000000FF");
        assert_eq!(hex_to_ass_color("00FF00"), "&H0000FF00");
        assert_eq!(hex_to_ass_color("0000FF"), "&H00FF0000");
        assert_eq!(hex_to_ass_color("FFFFFF"), "&H00FFFFFF");
    }

    #[test]
    fn test_hex_to_ass_color_with_hash() {
        assert_eq!(hex_to_ass_color("#FF0000"), "&H000000FF");
    }

    #[test]
    fn test_hex_to_ass_color_invalid() {
        assert_eq!(hex_to_ass_color("FFF"), "&H00FFFFFF");
        assert_eq!(hex_to_ass_color(""), "&H00FFFFFF");
    }

    #[test]
    fn test_format_ass_structure() {
        let entries = vec![
            TranslatedEntry {
                start_ms: 1000,
                end_ms: 4000,
                text: "Merhaba!".to_string(),
                speaker: Some("Character_A".to_string()),
                emotion: Some("happy".to_string()),
                color: Some("FF4444".to_string()),
            },
            TranslatedEntry {
                start_ms: 5000,
                end_ms: 8000,
                text: "Nasilsin?".to_string(),
                speaker: Some("Character_B".to_string()),
                emotion: Some("neutral".to_string()),
                color: Some("44AAFF".to_string()),
            },
        ];

        let ass = format_ass(&entries, "Test Title");

        assert!(ass.contains("[Script Info]"));
        assert!(ass.contains("Title: Test Title"));
        assert!(ass.contains("ScriptType: v4.00+"));
        assert!(ass.contains("[V4+ Styles]"));
        assert!(ass.contains("[Events]"));
        assert!(ass.contains("Merhaba!"));
        assert!(ass.contains("Nasilsin?"));
        assert!(ass.contains("Character_A"));
        assert!(ass.contains("Character_B"));
    }

    #[test]
    fn test_format_ass_default_style_for_no_speaker() {
        let entries = vec![TranslatedEntry {
            start_ms: 0,
            end_ms: 1000,
            text: "Test".to_string(),
            speaker: None,
            emotion: None,
            color: None,
        }];

        let ass = format_ass(&entries, "No Speaker");
        assert!(ass.contains("Style: Default,"));
        assert!(ass.contains("Dialogue: 0,0:00:00.00,0:00:01.00,Default,Default,0,0,0,,Test"));
    }

    #[test]
    fn test_format_ass_timing() {
        let entries = vec![TranslatedEntry {
            start_ms: 3661500,
            end_ms: 3665000,
            text: "Timed line".to_string(),
            speaker: None,
            emotion: None,
            color: None,
        }];

        let ass = format_ass(&entries, "Timing Test");
        assert!(ass.contains("1:01:01.50"));
        assert!(ass.contains("1:01:05.00"));
    }
}
