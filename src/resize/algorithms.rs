use std::f64::consts::PI;

use crate::ResizeAlgorithm;

pub struct Point;

impl ResizeAlgorithm for Point {
    #[inline(always)]
    fn support() -> u32 {
        0
    }

    #[inline(always)]
    fn new() -> Self {
        Self
    }

    #[inline(always)]
    fn process(&self, _x: f64) -> f64 {
        1.0_f64
    }
}

pub struct Bilinear;

impl ResizeAlgorithm for Bilinear {
    #[inline(always)]
    fn support() -> u32 {
        1
    }

    #[inline(always)]
    fn new() -> Self {
        Self
    }

    #[inline(always)]
    fn process(&self, x: f64) -> f64 {
        (1.0 - x.abs()).max(0.0)
    }
}

pub struct BicubicBSpline {
    polys: BicubicPolys,
}

impl ResizeAlgorithm for BicubicBSpline {
    #[inline(always)]
    fn support() -> u32 {
        2
    }

    #[inline(always)]
    fn new() -> Self {
        Self {
            polys: BicubicPolys::new(1.0, 0.0),
        }
    }

    #[inline(always)]
    fn process(&self, x: f64) -> f64 {
        self.polys.process(x)
    }
}

pub struct BicubicMitchell {
    polys: BicubicPolys,
}

impl ResizeAlgorithm for BicubicMitchell {
    #[inline(always)]
    fn support() -> u32 {
        2
    }

    #[inline(always)]
    fn new() -> Self {
        Self {
            polys: BicubicPolys::new(1.0 / 3.0, 1.0 / 3.0),
        }
    }

    #[inline(always)]
    fn process(&self, x: f64) -> f64 {
        self.polys.process(x)
    }
}

// (0, 1/2)
pub struct BicubicCatmullRom {
    polys: BicubicPolys,
}

impl ResizeAlgorithm for BicubicCatmullRom {
    #[inline(always)]
    fn support() -> u32 {
        2
    }

    #[inline(always)]
    fn new() -> Self {
        Self {
            polys: BicubicPolys::new(0.0, 0.5),
        }
    }

    #[inline(always)]
    fn process(&self, x: f64) -> f64 {
        self.polys.process(x)
    }
}

pub struct Lanczos3;

impl ResizeAlgorithm for Lanczos3 {
    #[inline(always)]
    fn support() -> u32 {
        3
    }

    #[inline(always)]
    fn new() -> Self {
        Self
    }

    #[inline(always)]
    fn process(&self, x: f64) -> f64 {
        lanczos_filter(x, 3)
    }
}

pub struct Lanczos4;

impl ResizeAlgorithm for Lanczos4 {
    #[inline(always)]
    fn support() -> u32 {
        4
    }

    #[inline(always)]
    fn new() -> Self {
        Self
    }

    #[inline(always)]
    fn process(&self, x: f64) -> f64 {
        lanczos_filter(x, 4)
    }
}

pub struct Spline16;

impl ResizeAlgorithm for Spline16 {
    #[inline(always)]
    fn support() -> u32 {
        2
    }

    #[inline(always)]
    fn new() -> Self {
        Self
    }

    #[inline(always)]
    fn process(&self, x: f64) -> f64 {
        let x = x.abs();

        if x < 1.0_f64 {
            return poly3(x, 1.0, -1.0 / 5.0, -9.0 / 5.0, 1.0);
        }
        if x < 2.0_f64 {
            let x = x - 1.0_f64;
            return poly3(x, 0.0, -7.0 / 15.0, 4.0 / 5.0, -1.0 / 3.0);
        }
        0.0_f64
    }
}

pub struct Spline36;

impl ResizeAlgorithm for Spline36 {
    #[inline(always)]
    fn support() -> u32 {
        3
    }

    #[inline(always)]
    fn new() -> Self {
        Self
    }

    #[inline(always)]
    fn process(&self, x: f64) -> f64 {
        let x = x.abs();

        if x < 1.0_f64 {
            return poly3(x, 1.0, -3.0 / 209.0, -453.0 / 209.0, 13.0 / 11.0);
        }
        if x < 2.0_f64 {
            let x = x - 1.0_f64;
            return poly3(x, 0.0, -156.0 / 209.0, 270.0 / 209.0, -6.0 / 11.0);
        }
        if x < 3.0_f64 {
            let x = x - 2.0_f64;
            return poly3(x, 0.0, 26.0 / 209.0, -45.0 / 209.0, 1.0 / 11.0);
        }
        0.0_f64
    }
}

pub struct Spline64;

impl ResizeAlgorithm for Spline64 {
    #[inline(always)]
    fn support() -> u32 {
        4
    }

    #[inline(always)]
    fn new() -> Self {
        Self
    }

    #[inline(always)]
    fn process(&self, x: f64) -> f64 {
        let x = x.abs();

        if x < 1.0_f64 {
            return poly3(x, 1.0, -3.0 / 2911.0, -6387.0 / 2911.0, 49.0 / 41.0);
        }
        if x < 2.0_f64 {
            let x = x - 1.0_f64;
            return poly3(x, 0.0, -2328.0 / 2911.0, 4032.0 / 2911.0, -24.0 / 41.0);
        }
        if x < 3.0_f64 {
            let x = x - 2.0_f64;
            return poly3(x, 0.0, 582.0 / 2911.0, -1008.0 / 2911.0, 6.0 / 41.0);
        }
        if x < 4.0_f64 {
            let x = x - 3.0_f64;
            return poly3(x, 0.0, -97.0 / 2911.0, 168.0 / 2911.0, -1.0 / 41.0);
        }
        0.0_f64
    }
}

#[derive(Clone, Copy)]
struct BicubicPolys {
    p0: f64,
    p2: f64,
    p3: f64,
    q0: f64,
    q1: f64,
    q2: f64,
    q3: f64,
}

impl BicubicPolys {
    pub fn new(b: f64, c: f64) -> Self {
        // p0: (  6.0 -  2.0 * b           ) / 6.0
        // p2: (-18.0 + 12.0 * b +  6.0 * c) / 6.0
        // p3: ( 12.0 -  9.0 * b -  6.0 * c) / 6.0
        // q0: (         8.0 * b + 24.0 * c) / 6.0
        // q1: (       -12.0 * b - 48.0 * c) / 6.0
        // q2: (         6.0 * b + 30.0 * c) / 6.0
        // q3: (              -b -  6.0 * c) / 6.0
        Self {
            p0: b.mul_add(-2.0, 6.0) / 6.0,
            p2: c.mul_add(6.0, b.mul_add(12.0, -18.0)) / 6.0,
            p3: c.mul_add(-6.0, b.mul_add(-9.0, 12.0)) / 6.0,
            q0: c.mul_add(24.0, b * 8.0) / 6.0,
            q1: c.mul_add(-48.0, b * -12.0) / 6.0,
            q2: c.mul_add(30.0, b * 6.0) / 6.0,
            q3: c.mul_add(-6.0, -b) / 6.0,
        }
    }

    #[inline(always)]
    pub fn process(&self, x: f64) -> f64 {
        let x = x.abs();

        if x < 1.0_f64 {
            return poly3(x, self.p0, 0.0, self.p2, self.p3);
        }
        if x < 2.0_f64 {
            return poly3(x, self.q0, self.q1, self.q2, self.q3);
        }
        0.0_f64
    }
}

#[inline(always)]
fn poly3(x: f64, c0: f64, c1: f64, c2: f64, c3: f64) -> f64 {
    // c0 + x * (c1 + x * (c2 + x * c3))
    x.mul_add(x.mul_add(x.mul_add(c3, c2), c1), c0)
}

#[inline(always)]
fn lanczos_filter(x: f64, taps: usize) -> f64 {
    let x = x.abs();
    if x < taps as f64 {
        sinc(x) * sinc(x / taps as f64)
    } else {
        0.0_f64
    }
}

#[inline(always)]
fn sinc(x: f64) -> f64 {
    // Guaranteed to not yield division by zero on IEEE machine with accurate sin(x).
    if x == 0.0_f64 {
        1.0_f64
    } else {
        (x * PI).sin() / (x * PI)
    }
}
