pub(crate) fn find_latest_checkpoint() -> Option<String> {
    use std::fs;
    use std::time::SystemTime;

    let search_patterns = [
        "./*.json",
        "./checkpoints/*.json",
        "./models/*.json",
        "./*.checkpoint.json",
    ];

    let mut candidates: Vec<(String, SystemTime)> = Vec::new();

    for pattern in &search_patterns {
        if let Ok(entries) = glob::glob(pattern) {
            for entry in entries.flatten() {
                if let Ok(metadata) = fs::metadata(&entry) {
                    if let Ok(modified) = metadata.modified() {
                        if let Ok(contents) = fs::read_to_string(&entry) {
                            if contents.contains("\"architecture\"")
                                && contents.contains("\"weights\"")
                            {
                                candidates.push((entry.to_string_lossy().to_string(), modified));
                            }
                        }
                    }
                }
            }
        }
    }

    if let Ok(entries) = fs::read_dir(".") {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().map_or(false, |e| e == "json") {
                if let Ok(metadata) = fs::metadata(&path) {
                    if let Ok(modified) = metadata.modified() {
                        if let Ok(contents) = fs::read_to_string(&path) {
                            if contents.contains("\"architecture\"")
                                && contents.contains("\"weights\"")
                            {
                                let path_str = path.to_string_lossy().to_string();
                                if !candidates.iter().any(|(p, _)| p == &path_str) {
                                    candidates.push((path_str, modified));
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    candidates.sort_by(|a, b| b.1.cmp(&a.1));
    candidates.into_iter().next().map(|(path, _)| path)
}
