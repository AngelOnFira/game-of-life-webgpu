//! Classic Conway's Game of Life patterns, as cell offsets from a pattern origin.
//!
//! Each pattern is a list of `(dx, dy)` cells; the app translates them onto the
//! grid by adding a stamp origin (centre of the grid by default) and produces
//! `(x, y, 1u32)` tuples for [`crate::gpu::Resources::stamp`].

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Pattern {
    /// 5 cells; travels diagonally with period 4.
    Glider,
    /// 36 cells; emits gliders periodically. Stable, classic.
    GosperGliderGun,
    /// 13 cells; oscillator with period 3.
    Pulsar,
    /// 5 cells; small methuselah that grows for ~1100 generations.
    RPentomino,
    /// 7 cells; methuselah that grows for ~5200 generations.
    Acorn,
}

impl Pattern {
    pub const ALL: &'static [Pattern] = &[
        Pattern::Glider,
        Pattern::GosperGliderGun,
        Pattern::Pulsar,
        Pattern::RPentomino,
        Pattern::Acorn,
    ];

    pub fn name(self) -> &'static str {
        match self {
            Pattern::Glider => "Glider",
            Pattern::GosperGliderGun => "Gosper glider gun",
            Pattern::Pulsar => "Pulsar (period 3)",
            Pattern::RPentomino => "R-pentomino",
            Pattern::Acorn => "Acorn",
        }
    }

    /// Cell offsets `(dx, dy)`. Pattern origin is the top-left of its bounding box.
    pub fn cells(self) -> &'static [(i32, i32)] {
        match self {
            Pattern::Glider => &[(1, 0), (2, 1), (0, 2), (1, 2), (2, 2)],

            // 36-cell Gosper glider gun. Bounding box 36×9.
            Pattern::GosperGliderGun => &[
                (24, 0),
                (22, 1),
                (24, 1),
                (12, 2),
                (13, 2),
                (20, 2),
                (21, 2),
                (34, 2),
                (35, 2),
                (11, 3),
                (15, 3),
                (20, 3),
                (21, 3),
                (34, 3),
                (35, 3),
                (0, 4),
                (1, 4),
                (10, 4),
                (16, 4),
                (20, 4),
                (21, 4),
                (0, 5),
                (1, 5),
                (10, 5),
                (14, 5),
                (16, 5),
                (17, 5),
                (22, 5),
                (24, 5),
                (10, 6),
                (16, 6),
                (24, 6),
                (11, 7),
                (15, 7),
                (12, 8),
                (13, 8),
            ],

            // 13×13 pulsar (period 3). Symmetric; here's the canonical layout.
            Pattern::Pulsar => &[
                (2, 0),
                (3, 0),
                (4, 0),
                (8, 0),
                (9, 0),
                (10, 0),
                (0, 2),
                (5, 2),
                (7, 2),
                (12, 2),
                (0, 3),
                (5, 3),
                (7, 3),
                (12, 3),
                (0, 4),
                (5, 4),
                (7, 4),
                (12, 4),
                (2, 5),
                (3, 5),
                (4, 5),
                (8, 5),
                (9, 5),
                (10, 5),
                (2, 7),
                (3, 7),
                (4, 7),
                (8, 7),
                (9, 7),
                (10, 7),
                (0, 8),
                (5, 8),
                (7, 8),
                (12, 8),
                (0, 9),
                (5, 9),
                (7, 9),
                (12, 9),
                (0, 10),
                (5, 10),
                (7, 10),
                (12, 10),
                (2, 12),
                (3, 12),
                (4, 12),
                (8, 12),
                (9, 12),
                (10, 12),
            ],

            Pattern::RPentomino => &[(1, 0), (2, 0), (0, 1), (1, 1), (1, 2)],

            Pattern::Acorn => &[(1, 0), (3, 1), (0, 2), (1, 2), (4, 2), (5, 2), (6, 2)],
        }
    }

    /// Bounding-box size used to centre the pattern.
    pub fn extent(self) -> (i32, i32) {
        let cells = self.cells();
        let mut w = 0;
        let mut h = 0;
        for &(x, y) in cells {
            if x > w {
                w = x;
            }
            if y > h {
                h = y;
            }
        }
        (w + 1, h + 1)
    }
}

/// Translate a pattern onto the grid centred at `(cx, cy)`. Returns
/// `(x, y, 1)` triples ready for `Resources::stamp`. Cells that fall outside
/// the grid are silently dropped.
pub fn stamp_centred(p: Pattern, cx: i32, cy: i32, grid_size: u32) -> Vec<(u32, u32, u32)> {
    let (w, h) = p.extent();
    let ox = cx - w / 2;
    let oy = cy - h / 2;
    p.cells()
        .iter()
        .filter_map(|&(dx, dy)| {
            let x = ox + dx;
            let y = oy + dy;
            if x < 0 || y < 0 || x >= grid_size as i32 || y >= grid_size as i32 {
                None
            } else {
                Some((x as u32, y as u32, 1))
            }
        })
        .collect()
}
