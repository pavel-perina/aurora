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
    sharpness: float      # exponent on the positive noise -> sparser, harder beams
    strength: float       # multiplicative brightening at full power
    white: float          # additive white fraction at full power
    fade_pow: float       # vertical fade exponent, beams live at the top
    shear: float = 0.15   # dx per dy (beam tilt), used when origin is None
    origin: tuple[float, float] | None = None   # ray vanishing point in
                          # normalized coords (y < 0 = above the frame);
                          # beams become a fan of rays from this point
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
        if c.origin is not None:
            # angle around a vanishing point above the frame -> diverging rays
            ox, oy = c.origin
            theta = np.arctan2(x / H - ox, y / H - oy)
            u = (theta - theta.min()) / (theta.max() - theta.min() + 1e-9)
        else:
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

@dataclass
class Preset:
    """Colors and intensity gains; the composition geometry is shared and
    lives in build_scene()."""
    ribbon: list[str]             # ribbon palette anchors, swept dark -> bright
    base: str                     # backdrop color behind the gradient blobs
    blobs: list[str]              # 5 colors: top-left, bottom-left, center,
                                  # top-right, bottom-right
    glow: str                     # horizon glow color
    glow_strength: float = 0.3
    wash: float = 1.0             # gain on the dense translucent sheet washes
    line: float = 1.0             # gain on the crisp hairlines
    curtain: float = 1.0          # gain on curtain ray brightening
    curtain_white: float = 1.0    # gain on additive white in the rays
    bloom_thresh: float = 0.55
    bloom_strength: float = 0.3
    exposure: float = 1.0
    saturate: float = 1.15


PRESETS = {
    "vista": Preset(
        ribbon=["#e8e03a", "#a8e83a", "#3ae87a", "#2ad8c8", "#27a0e8", "#8ad8ff"],
        base="#1e7f86",
        blobs=["#2e9e4f", "#e8d92e", "#1a8c96", "#2a6fa8", "#2596be"],
        glow="#fdfff0", glow_strength=0.35, line=1.1, saturate=1.22),
    "bronze": Preset(
        ribbon=["#7a3b10", "#b4651a", "#e89b2a", "#ffd980", "#fff3cf"],
        base="#4a2c14",
        blobs=["#3a2410", "#c98a2a", "#5a3618", "#2e1c0e", "#8a5a20"],
        glow="#ffe9b0", glow_strength=0.3, curtain=0.95, curtain_white=0.75,
        bloom_thresh=0.5),
    "ember": Preset(
        ribbon=["#8a200a", "#b8320c", "#ef7012", "#ffb13a", "#fff0c0"],
        base="#17181c",
        blobs=["#101014", "#56250c", "#1d1f26", "#26282e", "#3a1c10"],
        glow="#ff9a3a", glow_strength=0.18, wash=0.55, line=0.9,
        curtain=1.5, curtain_white=0.35, bloom_thresh=0.5,
        bloom_strength=0.25, saturate=1.1),
    "orchid": Preset(
        ribbon=["#6a2aa0", "#a44ae0", "#7a86e8", "#3ec4cf", "#eafaff"],
        base="#2c2347",
        blobs=["#45286a", "#8a3fae", "#33305e", "#1f5e74", "#2e8a96"],
        glow="#e8f6ff", glow_strength=0.28, line=1.05, curtain=0.9,
        curtain_white=0.6, bloom_thresh=0.5, saturate=1.18),
    "glacier": Preset(
        ribbon=["#1a3a8a", "#2a6fd0", "#3ab8e8", "#9ae8f0", "#f0fcff"],
        base="#0e1c30",
        blobs=["#11253f", "#1c4a6e", "#122a48", "#0b1626", "#1f5e80"],
        glow="#cfeeff", glow_strength=0.22, wash=0.7,
        curtain=1.3, curtain_white=0.5, bloom_thresh=0.5,
        saturate=1.12),
}


def build_scene(p: Preset, density: float = 1.0) -> Scene:
    """Shared composition: lower-left fan, pinch sheet on the right, faint
    upper back-sheet, long streaks along the bottom. `density` scales the
    hairline counts (1.0 = default, ~1.5 = busy, ~0.5 = sparse)."""
    def L(n: float) -> int:
        return max(2, int(round(n * density)))
    sheets = [
        # big airy fan rising from the lower left up to mid-frame
        Sheet(A=[(-0.10, 1.02), (0.45, 0.96), (0.95, 0.92), (1.65, 0.86)],
              B=[(-0.05, 0.20), (0.30, 0.52), (0.95, 0.78), (1.82, 0.48)],
              bow=0.06, bow1=-0.03, n_chords=1600, alpha=0.026 * p.wash,
              pal0=0.0, pal1=0.62, fade=0.12,
              lines=L(110), line_alpha=0.22 * p.line),
        # sheet sweeping right into a pinch (chords nearly cross -> caustic)
        Sheet(A=[(0.30, 1.06), (0.90, 0.90), (1.30, 0.76), (1.40, 0.52)],
              B=[(1.92, 0.98), (1.52, 0.88), (1.34, 0.78), (1.26, 0.88)],
              bow=-0.10, bow1=0.04, n_chords=1300, alpha=0.032 * p.wash,
              pal0=0.35, pal1=1.0, fade=0.10,
              lines=L(70), line_alpha=0.18 * p.line),
        # faint upper back-sheet
        Sheet(A=[(-0.15, 0.55), (0.30, 0.25), (0.90, 0.10), (1.60, 0.18)],
              B=[(-0.10, 0.95), (0.50, 0.60), (1.10, 0.40), (1.85, 0.30)],
              bow=0.05, n_chords=900, alpha=0.008 * p.wash,
              pal0=0.55, pal1=1.0, fade=0.2,
              lines=L(18), line_alpha=0.04 * p.line),
        # long shallow streaks hugging the bottom edge; guide curves run in
        # opposite directions so the chords cross mid-frame
        Sheet(A=[(-0.10, 0.86), (0.50, 0.92), (1.10, 0.98), (1.95, 0.96)],
              B=[(1.95, 1.06), (1.30, 1.02), (0.60, 1.04), (-0.10, 1.04)],
              bow=0.03, n_chords=700, alpha=0.005 * p.wash,
              pal0=0.10, pal1=0.80, fade=0.18,
              lines=L(45), line_alpha=0.09 * p.line),
        # narrow silk band climbing the left edge
        Sheet(A=[(-0.08, 0.95), (0.05, 0.75), (0.02, 0.50), (0.12, 0.28)],
              B=[(0.35, 1.00), (0.45, 0.75), (0.40, 0.45), (0.65, 0.15)],
              bow=0.05, n_chords=800, alpha=0.010 * p.wash,
              pal0=0.0, pal1=0.45, fade=0.15,
              lines=L(40), line_alpha=0.10 * p.line),
        # band rising from the pinch region along the right edge
        Sheet(A=[(1.30, 0.92), (1.45, 0.70), (1.55, 0.45), (1.60, 0.12)],
              B=[(1.95, 0.95), (1.85, 0.65), (1.92, 0.40), (1.75, 0.05)],
              bow=-0.04, n_chords=800, alpha=0.010 * p.wash,
              pal0=0.70, pal1=1.0, fade=0.15,
              lines=L(40), line_alpha=0.10 * p.line),
    ]
    blob_geo = [(0.15, 0.05, 0.45, 1.3), (0.05, 0.95, 0.40, 1.5),
                (1.05, 0.35, 0.60, 1.2), (1.80, 0.25, 0.55, 1.4),
                (1.85, 0.95, 0.45, 1.0)]
    return Scene(
        base=p.base,
        blobs=[Blob(x, y, s, c, w) for (x, y, s, w), c in zip(blob_geo, p.blobs)],
        curtains=[
            Curtain(n_knots=26, origin=(0.65, -0.55), sharpness=2.2,
                    strength=0.55 * p.curtain, white=0.04 * p.curtain_white,
                    fade_pow=1.6),
            Curtain(n_knots=70, origin=(1.05, -0.90), sharpness=3.5,
                    strength=0.30 * p.curtain, white=0.05 * p.curtain_white,
                    fade_pow=2.2, seed_off=7),
        ],
        ribbon_palette=Palette(p.ribbon), sheets=sheets,
        glow_y=1.07, glow_sigma=0.06, glow_color=p.glow,
        glow_strength=p.glow_strength,
        bloom_thresh=p.bloom_thresh, bloom_strength=p.bloom_strength,
        exposure=p.exposure, saturate=p.saturate)


# ---------------------------------------------------------------- main

def render(scene: Scene, W: int, H: int, ss: int, seed: int = 0) -> np.ndarray:
    rw, rh = W * ss, H * ss
    rng = np.random.default_rng(seed)
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
    ap.add_argument("--scene", default="vista", choices=sorted(PRESETS))
    ap.add_argument("--size", default="1920x1080")
    ap.add_argument("--ss", type=int, default=2, help="supersampling factor")
    ap.add_argument("--seed", type=int, default=0, help="varies the curtain rays")
    ap.add_argument("--density", type=float, default=1.0,
                    help="hairline density multiplier (0.5 sparse .. 1.5 busy)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--preview", action="store_true",
                    help="also write an 8-bit JPEG next to the PNG")
    args = ap.parse_args()

    W, H = (int(v) for v in args.size.split("x"))
    scene = build_scene(PRESETS[args.scene], density=args.density)
    out = args.out or f"aurora_{args.scene}_{W}x{H}.png"
    print(f"rendering {args.scene} {W}x{H} ss={args.ss}")
    img = render(scene, W, H, args.ss, args.seed)
    save16(img, out)
    print(f"wrote {out}")
    if args.preview:
        save_preview(img, out.rsplit(".", 1)[0] + ".jpg")


if __name__ == "__main__":
    main()
