//! Minimal parser for Conway's Game of Life RLE (Run-Length Encoded) patterns.
//!
//! RLE is the de-facto pattern exchange format used on
//! [LifeWiki](https://conwaylife.com/wiki/Run_Length_Encoded). Examples
//! ship with most Life implementations and most published patterns.
//!
//! Format (abridged):
//! - Lines starting with `#` are comments / metadata.
//! - The first non-comment line is the header: `x = W, y = H, rule = B3/S23`.
//! - The body uses run-length encoding: `N b` = N dead cells, `N o` = N alive
//!   cells, `N$` = N line ends, `!` ends the pattern. A missing N means 1.
//! - The body is whitespace-tolerant and may span any number of lines.
//!
//! We only support the standard `B3/S23` rule. Anything else is rejected.

/// Parsed pattern: a list of alive-cell offsets `(x, y)` plus the declared
/// bounding box.
#[derive(Debug, Clone)]
pub struct RlePattern {
    pub width: u32,
    pub height: u32,
    pub cells: Vec<(u32, u32)>,
}

#[derive(Debug, thiserror::Error)]
pub enum RleError {
    #[error("missing or malformed RLE header line (expected `x = W, y = H, rule = …`)")]
    MissingHeader,
    #[error("unsupported rule `{0}` — this demo only runs B3/S23")]
    UnsupportedRule(String),
    #[error("unexpected character `{0}` in RLE body")]
    UnexpectedChar(char),
    #[error("pattern declared {declared}×{declared_h} but body produced cell at ({x},{y})", declared = .declared_w, declared_h = .declared_h)]
    OutOfBounds {
        declared_w: u32,
        declared_h: u32,
        x: u32,
        y: u32,
    },
}

impl RlePattern {
    pub fn parse(text: &str) -> Result<Self, RleError> {
        // 1. Find header (first non-comment, non-blank line).
        let mut lines = text.lines().map(str::trim);
        let header = loop {
            let line = lines.next().ok_or(RleError::MissingHeader)?;
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            break line;
        };

        // 2. Parse header. Format: `x = W, y = H[, rule = R]`.
        let mut width = 0u32;
        let mut height = 0u32;
        let mut rule: Option<String> = None;
        for kv in header.split(',') {
            let mut split = kv.split('=');
            let k = split.next().unwrap_or("").trim();
            let v = split.next().unwrap_or("").trim();
            match k {
                "x" => width = v.parse().map_err(|_| RleError::MissingHeader)?,
                "y" => height = v.parse().map_err(|_| RleError::MissingHeader)?,
                "rule" => rule = Some(v.to_string()),
                _ => {}
            }
        }
        if width == 0 || height == 0 {
            return Err(RleError::MissingHeader);
        }
        if let Some(r) = rule.as_deref()
            && !is_b3s23(r)
        {
            return Err(RleError::UnsupportedRule(r.to_string()));
        }

        // 3. Parse body. Concatenate remaining lines, then walk a small state
        // machine: digits accumulate `run`, letters/`$`/`!` apply it.
        let mut body = String::new();
        for line in lines {
            body.push_str(line);
        }
        let mut cells = Vec::new();
        let mut run: u32 = 0;
        let mut x: u32 = 0;
        let mut y: u32 = 0;
        for ch in body.chars() {
            match ch {
                '0'..='9' => run = run * 10 + (ch as u32 - '0' as u32),
                'b' | 'B' => {
                    x += run.max(1);
                    run = 0;
                }
                'o' | 'O' => {
                    let n = run.max(1);
                    for i in 0..n {
                        let px = x + i;
                        if px >= width || y >= height {
                            return Err(RleError::OutOfBounds {
                                declared_w: width,
                                declared_h: height,
                                x: px,
                                y,
                            });
                        }
                        cells.push((px, y));
                    }
                    x += n;
                    run = 0;
                }
                '$' => {
                    y += run.max(1);
                    x = 0;
                    run = 0;
                }
                '!' => break,
                ch if ch.is_whitespace() => {}
                ch => return Err(RleError::UnexpectedChar(ch)),
            }
        }

        Ok(Self {
            width,
            height,
            cells,
        })
    }

    /// Translate the cells so the pattern is centred on `(cx, cy)` on a
    /// `grid_size × grid_size` grid; out-of-bounds cells are dropped.
    /// Output is in the `(x, y, alive)` shape used by `GolEngine::stamp`.
    pub fn stamp_centred(&self, cx: i32, cy: i32, grid_size: u32) -> Vec<(u32, u32, u32)> {
        let ox = cx - self.width as i32 / 2;
        let oy = cy - self.height as i32 / 2;
        self.cells
            .iter()
            .filter_map(|&(dx, dy)| {
                let x = ox + dx as i32;
                let y = oy + dy as i32;
                if x < 0 || y < 0 || x >= grid_size as i32 || y >= grid_size as i32 {
                    None
                } else {
                    Some((x as u32, y as u32, 1))
                }
            })
            .collect()
    }
}

fn is_b3s23(rule: &str) -> bool {
    // Normalise: strip whitespace, lowercase.
    let r: String = rule.chars().filter(|c| !c.is_whitespace()).collect();
    let r = r.to_ascii_lowercase();
    matches!(r.as_str(), "b3/s23" | "23/3")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_glider() {
        // Standard glider in RLE.
        let p = RlePattern::parse(
            "#N Glider\nx = 3, y = 3, rule = B3/S23\nbob$2bo$3o!",
        )
        .unwrap();
        assert_eq!(p.width, 3);
        assert_eq!(p.height, 3);
        // 5 alive cells: (1,0), (2,1), (0,2), (1,2), (2,2).
        let mut sorted = p.cells.clone();
        sorted.sort();
        assert_eq!(
            sorted,
            vec![(0, 2), (1, 0), (1, 2), (2, 1), (2, 2)]
        );
    }

    #[test]
    fn parse_blinker_with_runs() {
        let p = RlePattern::parse("x = 3, y = 1, rule = B3/S23\n3o!").unwrap();
        assert_eq!(p.cells, vec![(0, 0), (1, 0), (2, 0)]);
    }

    #[test]
    fn rejects_unsupported_rule() {
        let err = RlePattern::parse("x = 1, y = 1, rule = B36/S23\no!").unwrap_err();
        assert!(matches!(err, RleError::UnsupportedRule(_)));
    }
}
