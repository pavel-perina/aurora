#!/usr/bin/env python3
"""Vista-Aurora-style wallpaper generator.

Layers (back to front), all composited in linear RGB float32:
  1. background  -- smooth multi-blob color gradient (RBF mix)
  2. curtains    -- broad soft vertical light shafts (smooth 1-D noise, sheared)
  3. ribbons     -- string-art chord families between two Bezier guide curves,
                    additively splatted; envelopes/caustics emerge automatically
  4. bloom + horizon glow
Output: 16-bit PNG (sRGB-encoded).
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass, field

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter, gaussian_filter1d

# ---------------------------------------------------------------- color

def srgb_to_linear(c: np.ndarray) -> np.ndarray:
    c = np.asarray(c, dtype=np.float64)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(c: np.ndarray) -> np.ndarray:
    c = np.clip(c, 0.0, 1.0)
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * c ** (1 / 2.4) - 0.055)


def hex_rgb(h: str) -> np.ndarray:
    h = h.lstrip("#")
    return np.array([int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4)])


_LRGB_TO_LMS = np.array([
    [0.4122214708, 0.5363325363, 0.0514459929],
    [0.2119034982, 0.6806995451, 0.1073969566],
    [0.0883024619, 0.2817188376, 0.6299787005]])
_LMS_TO_OKLAB = np.array([
    [0.2104542553, 0.7936177850, -0.0040720468],
    [1.9779984951, -2.4285922050, 0.4505937099],
    [0.0259040371, 0.7827717662, -0.8086757660]])


def lin_to_oklab(rgb: np.ndarray) -> np.ndarray:
    lms = rgb @ _LRGB_TO_LMS.T
    return np.cbrt(lms) @ _LMS_TO_OKLAB.T


def oklab_to_lin(lab: np.ndarray) -> np.ndarray:
    lms = (lab @ np.linalg.inv(_LMS_TO_OKLAB).T) ** 3
    return lms @ np.linalg.inv(_LRGB_TO_LMS).T


class Palette:
    """Piecewise-linear ramp through sRGB anchor colors, interpolated in Oklab."""

    def __init__(self, anchors: list[str]):
        lab = lin_to_oklab(srgb_to_linear(np.array([hex_rgb(a) for a in anchors])))
        self.pos = np.linspace(0.0, 1.0, len(anchors))
        self.lab = lab

    def __call__(self, t: np.ndarray) -> np.ndarray:
        t = np.clip(np.asarray(t, dtype=np.float64), 0.0, 1.0)
        out = np.empty(t.shape + (3,))
        for c in range(3):
            out[..., c] = np.interp(t, self.pos, self.lab[:, c])
        return np.clip(oklab_to_lin(out), 0.0, None)


# ---------------------------------------------------------------- scene types

@dataclass
class Blob:
    """One soft color source for the background RBF mix. Coords normalized:
    x in [0, aspect], y in [0, 1], y down."""
    x: float
    y: float
    sigma: float
    color: str
    weight: float = 1.0


@dataclass
class Curtain:
    n_knots: int          # knots of the random brightness profile across x
    shear: float          # dx per dy (beam tilt)
    sharpness: float      # exponent on the positive noise -> sparser, harder beams
    strength: float       # multiplicative brightening at full power
    white: float          # additive white fraction at full power
    fade_pow: float       # vertical fade exponent, beams live at the top
    smooth: float = 1.5   # profile smoothing in knot units
    seed_off: int = 0


@dataclass
class Sheet:
    """Chord family between cubic Beziers A(t) and B(t); chords are quadratic
    Beziers bowed via a moving control point."""
    A: list[tuple[float, float]]          # 4 control points, normalized coords
    B: list[tuple[float, float]]
    bow: float = 0.0                      # chord bow, fraction of chord length
    bow1: float | None = None             # bow at t=1 (lerp from bow), optional
    n_chords: int = 1400
    alpha: float = 6.0                    # total energy of the whole sheet
    pal0: float = 0.0                     # palette range swept over t
    pal1: float = 1.0
    fade: float = 0.15                    # soft fade of chord ends (s near 0/1)
    gamma_t: float = 1.0                  # chord density bias along t
    lines: int = 0                        # sparse crisp lines on top of the sheet
    line_alpha: float = 0.06


@dataclass
class Scene:
    base: str                             # backdrop color behind the blobs
    blobs: list[Blob]
    curtains: list[Curtain]
    ribbon_palette: Palette
    sheets: list[Sheet]
    glow_y: float = 1.0                   # horizon glow center (normalized y)
    glow_sigma: float = 0.08
    glow_color: str = "#ffffff"
    glow_strength: float = 0.55
    bloom_thresh: float = 0.55
    bloom_strength: float = 0.35
    exposure: float = 1.0
    saturate: float = 1.0                 # Oklab chroma scale on the final image


# ---------------------------------------------------------------- layers

def bezier(pts: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Cubic Bezier, pts (4,2), t (N,) -> (N,2)."""
    t = t[:, None]
    u = 1.0 - t
    return (u ** 3 * pts[0] + 3 * u ** 2 * t * pts[1]
            + 3 * u * t ** 2 * pts[2] + t ** 3 * pts[3])


def render_background(scene: Scene, W: int, H: int) -> np.ndarray:
    aspect = W / H
    y, x = np.mgrid[0:H, 0:W].astype(np.float32)
    u = x / H            # so u spans [0, aspect], square units
    v = y / H * (H / H)
    v = y / H
    wsum = np.full((H, W), 1e-4, dtype=np.float32)
    csum = np.empty((H, W, 3), dtype=np.float32)
    base = srgb_to_linear(hex_rgb(scene.base)).astype(np.float32)
    csum[:] = base * 1e-4
    for b in scene.blobs:
        d2 = (u - b.x) ** 2 + (v - b.y) ** 2
        w = (b.weight * np.exp(-d2 / (2 * b.sigma ** 2))).astype(np.float32)
        col = srgb_to_linear(hex_rgb(b.color)).astype(np.float32)
        wsum += w
        csum += w[..., None] * col
    return csum / wsum[..., None]


def render_curtains(scene: Scene, W: int, H: int, rng: np.random.Generator,
                    bg: np.ndarray) -> np.ndarray:
    y, x = np.mgrid[0:H, 0:W].astype(np.float32)
    v = y / H
    add = np.zeros((H, W, 3), dtype=np.float32)
    white = np.float32(1.0)
    for c in scene.curtains:
        r = np.random.default_rng(rng.integers(1 << 31) + c.seed_off)
        # smooth positive 1-D profile over a sheared horizontal coordinate
        knots = r.normal(0, 1, c.n_knots)
        knots = gaussian_filter1d(knots, c.smooth, mode="wrap")
        prof_x = np.linspace(0, 1, 4096)
        kk = np.concatenate([knots, knots[:1]])          # periodic profile
        prof = np.interp(prof_x * c.n_knots, np.arange(c.n_knots + 1), kk)
        prof = gaussian_filter1d(prof, 4096 / c.n_knots * 0.35, mode="wrap")
        prof = np.maximum(prof, 0.0)
        if prof.max() > 0:
            prof /= prof.max()
        prof = prof ** c.sharpness
        u = (x + c.shear * y) / W
        u -= np.floor(u)
        m = np.interp(u.ravel(), prof_x, prof).reshape(H, W).astype(np.float32)
        m *= (1.0 - v) ** c.fade_pow
        add += bg * (c.strength * m)[..., None]
        add += (c.white * m)[..., None] * white
    return add


def render_sheets(scene: Scene, W: int, H: int, ss: int) -> np.ndarray:
    acc = np.zeros((H, W, 3), dtype=np.float64)
    pal = scene.ribbon_palette
    passes = []
    for sh in scene.sheets:
        passes.append((sh, sh.n_chords, sh.alpha))
        if sh.lines > 0:
            passes.append((sh, sh.lines, sh.line_alpha))
    for sh, n_chords, alpha in passes:
        t = np.linspace(0.0, 1.0, n_chords) ** sh.gamma_t
        A = bezier(np.array(sh.A) * H, t)            # px coords
        B = bezier(np.array(sh.B) * H, t)
        chord = B - A
        clen = np.hypot(chord[:, 0], chord[:, 1])
        perp = np.stack([-chord[:, 1], chord[:, 0]], axis=1) / (clen[:, None] + 1e-9)
        bow1 = sh.bow if sh.bow1 is None else sh.bow1
        bows = (sh.bow + (bow1 - sh.bow) * t)[:, None]
        C = 0.5 * (A + B) + perp * bows * clen[:, None]

        M = max(64, int(clen.max() * 1.5))
        s = np.linspace(0.0, 1.0, M, dtype=np.float64)[None, :, None]
        P = ((1 - s) ** 2 * A[:, None, :] + 2 * s * (1 - s) * C[:, None, :]
             + s ** 2 * B[:, None, :])              # (N, M, 2)

        col = pal(sh.pal0 + (sh.pal1 - sh.pal0) * t)  # (N,3)
        # alpha = brightness a single chord adds per pixel it crosses;
        # sample spacing is clen/M px, so weight = alpha * clen / M
        # (times ss: downsampling averages the thin line over ss^2 subpixels)
        wt = np.broadcast_to((alpha * ss * clen / M)[:, None],
                             (n_chords, M)).copy()
        if sh.fade > 0:
            sv = s[0, :, 0]
            wt *= np.clip(sv / sh.fade, 0, 1) * np.clip((1 - sv) / sh.fade, 0, 1)
        wrgb = wt[..., None] * col[:, None, :]        # (N,M,3)

        px = P[..., 0].ravel()
        py = P[..., 1].ravel()
        wrgb = wrgb.reshape(-1, 3)
        ok = (px >= 0) & (px < W - 1) & (py >= 0) & (py < H - 1)
        px, py, wrgb = px[ok], py[ok], wrgb[ok]
        x0 = px.astype(np.int64)
        y0 = py.astype(np.int64)
        fx, fy = px - x0, py - y0
        flat = acc.reshape(-1, 3)
        n = W * H
        for ix, iy, w in ((x0, y0, (1 - fx) * (1 - fy)), (x0 + 1, y0, fx * (1 - fy)),
                          (x0, y0 + 1, (1 - fx) * fy), (x0 + 1, y0 + 1, fx * fy)):
            idx = iy * W + ix
            for ch in range(3):
                flat[:, ch] += np.bincount(idx, weights=w * wrgb[:, ch], minlength=n)
    return acc.astype(np.float32)


def add_glow_and_bloom(img: np.ndarray, scene: Scene, W: int, H: int) -> np.ndarray:
    y = (np.arange(H, dtype=np.float32) / H)[:, None]
    band = np.exp(-0.5 * ((y - scene.glow_y) / scene.glow_sigma) ** 2)
    gcol = srgb_to_linear(hex_rgb(scene.glow_color)).astype(np.float32)
    img = img + (scene.glow_strength * band)[..., None] * gcol[None, None, :]

    # clamp so HDR caustic spikes bloom gently instead of nuking the frame
    hi = np.minimum(np.maximum(img - scene.bloom_thresh, 0.0).max(axis=2), 1.5)
    s1, s2 = 0.004 * W, 0.018 * W
    bloom = gaussian_filter(hi, s1) + 0.6 * gaussian_filter(hi, s2)
    img = img + scene.bloom_strength * bloom[..., None]
    return img


def tonemap(img: np.ndarray, exposure: float, knee: float = 0.75) -> np.ndarray:
    """Linear below the knee, smooth exponential roll-off into white above."""
    img = img * exposure
    over = img > knee
    soft = knee + (1.0 - knee) * (1.0 - np.exp(-(img - knee) / (1.0 - knee)))
    return np.where(over, soft, img)


# ---------------------------------------------------------------- scenes

def scene_vista(seed: int) -> Scene:
    rng = np.random.default_rng(seed)
    pal = Palette(["#e8e03a", "#a8e83a", "#3ae87a", "#2ad8c8", "#27a0e8", "#8ad8ff"])
    sheets = [
        # big airy fan rising from the lower left
        Sheet(A=[(-0.10, 1.02), (0.45, 0.96), (0.95, 0.92), (1.65, 0.86)],
              B=[(-0.05, 0.30), (0.25, 0.75), (0.90, 0.95), (1.78, 0.66)],
              bow=0.06, bow1=-0.03, n_chords=1600, alpha=0.020,
              pal0=0.0, pal1=0.62, fade=0.12, lines=80, line_alpha=0.20),
        # sheet sweeping right into a pinch (chords nearly cross -> caustic)
        Sheet(A=[(0.30, 1.08), (0.95, 0.93), (1.35, 0.82), (1.45, 0.62)],
              B=[(1.90, 1.05), (1.55, 0.95), (1.38, 0.86), (1.30, 0.94)],
              bow=-0.10, bow1=0.04, n_chords=1300, alpha=0.030,
              pal0=0.35, pal1=1.0, fade=0.10, lines=55, line_alpha=0.16),
        # faint upper back-sheet
        Sheet(A=[(-0.15, 0.55), (0.30, 0.25), (0.90, 0.10), (1.60, 0.18)],
              B=[(-0.10, 0.95), (0.50, 0.60), (1.10, 0.40), (1.85, 0.30)],
              bow=0.05, n_chords=900, alpha=0.008,
              pal0=0.55, pal1=1.0, fade=0.2),
    ]
    return Scene(
        base="#1e7f86",
        blobs=[
            Blob(0.15, 0.05, 0.45, "#2e9e4f", 1.3),
            Blob(0.05, 0.95, 0.40, "#e8d92e", 1.5),
            Blob(1.05, 0.35, 0.60, "#1a8c96", 1.2),
            Blob(1.80, 0.25, 0.55, "#2a6fa8", 1.4),
            Blob(1.85, 0.95, 0.45, "#2596be", 1.0),
        ],
        curtains=[
            Curtain(n_knots=22, shear=0.18, sharpness=2.2, strength=0.55,
                    white=0.04, fade_pow=1.6),
            Curtain(n_knots=60, shear=0.10, sharpness=3.5, strength=0.30,
                    white=0.05, fade_pow=2.2, seed_off=7),
        ],
        ribbon_palette=pal, sheets=sheets,
        glow_y=1.07, glow_sigma=0.06, glow_color="#fdfff0", glow_strength=0.35,
        bloom_thresh=0.55, bloom_strength=0.3, exposure=1.0, saturate=1.22)


def scene_bronze(seed: int) -> Scene:
    rng = np.random.default_rng(seed)
    pal = Palette(["#7a3b10", "#b4651a", "#e89b2a", "#ffd980", "#fff3cf"])
    sheets = [
        Sheet(A=[(-0.10, 1.02), (0.45, 0.96), (0.95, 0.92), (1.65, 0.86)],
              B=[(-0.05, 0.30), (0.25, 0.75), (0.90, 0.95), (1.78, 0.66)],
              bow=0.06, bow1=-0.03, n_chords=1600, alpha=0.025,
              pal0=0.25, pal1=0.95, fade=0.12, lines=80, line_alpha=0.18),
        Sheet(A=[(0.30, 1.08), (0.95, 0.93), (1.35, 0.82), (1.45, 0.62)],
              B=[(1.90, 1.05), (1.55, 0.95), (1.38, 0.86), (1.30, 0.94)],
              bow=-0.10, bow1=0.04, n_chords=1300, alpha=0.028,
              pal0=0.45, pal1=1.0, fade=0.10, lines=55, line_alpha=0.14),
        Sheet(A=[(-0.15, 0.55), (0.30, 0.25), (0.90, 0.10), (1.60, 0.18)],
              B=[(-0.10, 0.95), (0.50, 0.60), (1.10, 0.40), (1.85, 0.30)],
              bow=0.05, n_chords=900, alpha=0.007,
              pal0=0.4, pal1=0.9, fade=0.2),
    ]
    return Scene(
        base="#4a2c14",
        blobs=[
            Blob(0.15, 0.05, 0.45, "#3a2410", 1.3),
            Blob(0.05, 0.95, 0.40, "#c98a2a", 1.5),
            Blob(1.05, 0.35, 0.60, "#5a3618", 1.2),
            Blob(1.80, 0.25, 0.55, "#2e1c0e", 1.4),
            Blob(1.85, 0.95, 0.45, "#8a5a20", 1.0),
        ],
        curtains=[
            Curtain(n_knots=22, shear=0.18, sharpness=2.2, strength=0.50,
                    white=0.03, fade_pow=1.6),
            Curtain(n_knots=60, shear=0.10, sharpness=3.5, strength=0.30,
                    white=0.03, fade_pow=2.2, seed_off=7),
        ],
        ribbon_palette=pal, sheets=sheets,
        glow_y=1.07, glow_sigma=0.06, glow_color="#ffe9b0", glow_strength=0.3,
        bloom_thresh=0.5, bloom_strength=0.3, exposure=1.0, saturate=1.15)


def scene_ember(seed: int) -> Scene:
    rng = np.random.default_rng(seed)
    pal = Palette(["#5a0e08", "#a82a0c", "#e8650f", "#ffa62a", "#ffe9a8"])
    sheets = [
        Sheet(A=[(-0.10, 1.02), (0.45, 0.96), (0.95, 0.92), (1.65, 0.86)],
              B=[(-0.05, 0.30), (0.25, 0.75), (0.90, 0.95), (1.78, 0.66)],
              bow=0.06, bow1=-0.03, n_chords=1600, alpha=0.022,
              pal0=0.3, pal1=0.9, fade=0.12, lines=80, line_alpha=0.16),
        Sheet(A=[(0.30, 1.08), (0.95, 0.93), (1.35, 0.82), (1.45, 0.62)],
              B=[(1.90, 1.05), (1.55, 0.95), (1.38, 0.86), (1.30, 0.94)],
              bow=-0.10, bow1=0.04, n_chords=1300, alpha=0.025,
              pal0=0.45, pal1=1.0, fade=0.10, lines=55, line_alpha=0.14),
        Sheet(A=[(-0.15, 0.55), (0.30, 0.25), (0.90, 0.10), (1.60, 0.18)],
              B=[(-0.10, 0.95), (0.50, 0.60), (1.10, 0.40), (1.85, 0.30)],
              bow=0.05, n_chords=900, alpha=0.006,
              pal0=0.3, pal1=0.8, fade=0.2),
    ]
    return Scene(
        base="#17181c",
        blobs=[
            Blob(0.15, 0.05, 0.45, "#101014", 1.3),
            Blob(0.05, 0.95, 0.40, "#56250c", 1.5),
            Blob(1.05, 0.35, 0.60, "#1d1f26", 1.2),
            Blob(1.80, 0.25, 0.55, "#26282e", 1.4),
            Blob(1.85, 0.95, 0.45, "#3a1c10", 1.0),
        ],
        curtains=[
            Curtain(n_knots=22, shear=0.18, sharpness=2.2, strength=0.8,
                    white=0.015, fade_pow=1.6),
            Curtain(n_knots=60, shear=0.10, sharpness=3.5, strength=0.5,
                    white=0.015, fade_pow=2.2, seed_off=7),
        ],
        ribbon_palette=pal, sheets=sheets,
        glow_y=1.07, glow_sigma=0.06, glow_color="#ff9a3a", glow_strength=0.18,
        bloom_thresh=0.5, bloom_strength=0.25, exposure=1.0, saturate=1.1)


SCENES = {"vista": scene_vista, "bronze": scene_bronze, "ember": scene_ember}


# ---------------------------------------------------------------- main

def render(scene: Scene, W: int, H: int, ss: int) -> np.ndarray:
    rw, rh = W * ss, H * ss
    rng = np.random.default_rng(0)
    t0 = time.time()
    img = render_background(scene, rw, rh)
    img += render_curtains(scene, rw, rh, rng, img)
    print(f"  bg+curtains  {time.time() - t0:6.1f}s")
    img += render_sheets(scene, rw, rh, ss)
    print(f"  +sheets      {time.time() - t0:6.1f}s")
    img = add_glow_and_bloom(img, scene, rw, rh)
    img = tonemap(img, scene.exposure)
    if scene.saturate != 1.0:
        lab = lin_to_oklab(np.clip(img, 0.0, 1.0))
        lab[..., 1:] *= scene.saturate
        img = np.clip(oklab_to_lin(lab), 0.0, 1.0).astype(np.float32)
    print(f"  +glow/tone   {time.time() - t0:6.1f}s")
    if ss > 1:
        img = img.reshape(H, ss, W, ss, 3).mean(axis=(1, 3))
    return img


def save16(img: np.ndarray, path: str) -> None:
    out = np.round(linear_to_srgb(img) * 65535.0).astype(np.uint16)
    cv2.imwrite(path, out[..., ::-1])  # RGB -> BGR


def save_preview(img: np.ndarray, path: str) -> None:
    out = np.round(linear_to_srgb(img) * 255.0).astype(np.uint8)
    cv2.imwrite(path, out[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, 92])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scene", default="vista", choices=sorted(SCENES))
    ap.add_argument("--size", default="1920x1080")
    ap.add_argument("--ss", type=int, default=2, help="supersampling factor")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--preview", action="store_true",
                    help="also write an 8-bit JPEG next to the PNG")
    args = ap.parse_args()

    W, H = (int(v) for v in args.size.split("x"))
    scene = SCENES[args.scene](args.seed)
    out = args.out or f"aurora_{args.scene}_{W}x{H}.png"
    print(f"rendering {args.scene} {W}x{H} ss={args.ss}")
    img = render(scene, W, H, args.ss)
    save16(img, out)
    print(f"wrote {out}")
    if args.preview:
        save_preview(img, out.rsplit(".", 1)[0] + ".jpg")


if __name__ == "__main__":
    main()
