#!/usr/bin/env python3
"""Calibrate the Daheng's intrinsics from a checkerboard, and say how much to trust them.

    python scripts/deploy/dahengcal.py --square-mm 25
    python scripts/deploy/dahengcal.py --square-mm 25 --pattern 9x6 --views 20

Writes `deploy/calib/daheng_intrinsics.json`, which `run.py`, `extcal.py` and
`preflight.py` read.

Unlike a RealSense, this camera ships **no intrinsics at all** -- there is no lens
until someone screws one on, so `fx` is a property of the pairing and has to be
measured.  Everything downstream rests on it: `fx` sets the pixel span of a cube
tile, which is the whole reason for moving to this camera, and a systematic error
in `fx` biases every recovered distance by the same ratio.

**A calibration reports its own error, and that number is not a validation.**  The
RMS that `calibrateCamera` returns is the residual of a fit to the very points
that determined the parameters; it falls as you add distortion terms whether or
not they describe the lens.  So this holds views back: it fits on one subset and
reports reprojection on the other, and prints both.  A held-out error much worse
than the fit error means the model is chasing noise, most often because the views
are all alike.

Two ways a checkerboard calibration goes quietly wrong, both reported here:

*   **Views that do not differ.**  Board poses clustered in one place and one
    orientation leave `fx` and the board distance trading off against each other.
    Corner coverage and the spread of apparent board scale are printed so a thin
    dataset is visible rather than merely producing a confident-looking number.
*   **A wrong square size.**  Nothing in the fit can detect it: every recovered
    distance, and `fx` with it, scales by exactly the ratio.  Measure the printed
    square with calipers and pass the measurement.

The board's inner-corner count is detected on the first accepted view unless
`--pattern` is given.  A partly framed board yields a sub-grid, which is still a
valid calibration target -- but the count must then be the same in every view, so
the detected one is locked in and views that do not match it are refused.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from dexrobot021_mjlab.paths import calib_dir  # noqa: E402


def detect(image, pattern):
    import cv2

    flags = cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY
    ok, corners = cv2.findChessboardCornersSB(image, pattern, flags)
    return (corners.reshape(-1, 2).astype(np.float64) if ok else None)


def autodetect_pattern(image, lo=4, hi=16):
    """Largest inner-corner grid the board shows.  Locked in for the whole run."""
    import cv2

    best = None
    for cols in range(lo, hi):
        for rows in range(3, cols + 1):
            ok, _ = cv2.findChessboardCornersSB(image, (cols, rows),
                                                cv2.CALIB_CB_EXHAUSTIVE)
            if ok and (best is None or cols * rows > best[0] * best[1]):
                best = (cols, rows)
    return best


def object_points(pattern, square_m):
    # float32, not float64: `calibrateCamera` wants Point3f and rejects the
    # double array with "objectPoints should contain vector of vectors of points
    # of type Point3f", which it raises AFTER the whole collection is in hand.
    cols, rows = pattern
    grid = np.zeros((rows * cols, 3), np.float32)
    grid[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    return grid * square_m


def novelty(corners, accepted, width):
    """Smallest mean corner displacement against any accepted view, in pixels."""
    if not accepted:
        return float("inf")
    return min(float(np.mean(np.linalg.norm(corners - prev, axis=1)))
               for prev in accepted)


def coverage(all_corners, shape, bins=8):
    """Fraction of an 8x8 image grid that any detected corner has landed in."""
    hit = np.zeros((bins, bins), bool)
    h, w = shape
    for corners in all_corners:
        xs = np.clip((corners[:, 0] / w * bins).astype(int), 0, bins - 1)
        ys = np.clip((corners[:, 1] / h * bins).astype(int), 0, bins - 1)
        hit[ys, xs] = True
    return float(hit.mean())


def board_scale(corners, pattern):
    """Median spacing between horizontally adjacent corners, in pixels."""
    cols = pattern[0]
    grid = corners.reshape(pattern[1], cols, 2)
    return float(np.median(np.linalg.norm(np.diff(grid, axis=1), axis=2)))


def fit_and_report(accepted, pattern, shape, args) -> int:
    import cv2

    from dexrobot021_mjlab.deploy.camera import DahengSource

    if len(accepted) < 6:
        raise SystemExit(f"only {len(accepted)} views -- need at least 6")

    # Save the corners before fitting anything: collection costs the operator
    # minutes, the fit milliseconds, and `--from-npz` refits without recollecting.
    corners_npz = args.out.with_suffix(".corners.npz")
    corners_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(corners_npz, corners=np.stack(accepted),
                        pattern=np.array(pattern), square_mm=args.square_mm,
                        width=shape[1], height=shape[0])
    print(f"\nsaved {len(accepted)} views to {corners_npz}"
          f"  (refit with --from-npz {corners_npz})")

    objp = object_points(pattern, args.square_mm / 1000.0)
    obj = [objp for _ in accepted]
    img = [c.reshape(-1, 1, 2).astype(np.float32) for c in accepted]
    size = (shape[1], shape[0])

    order = np.random.default_rng(0).permutation(len(accepted))
    n_fit = max(5, int(round(len(accepted) * (1.0 - args.holdout))))
    fit, held = order[:n_fit], order[n_fit:]

    flags = {
        "k1": cv2.CALIB_FIX_K2 | cv2.CALIB_FIX_K3 | cv2.CALIB_ZERO_TANGENT_DIST,
        "k1k2": cv2.CALIB_FIX_K3 | cv2.CALIB_ZERO_TANGENT_DIST,
        "k1k2p": cv2.CALIB_FIX_K3,
        "full": 0,
    }[args.dist_model]
    rms, K, dist, _, _ = cv2.calibrateCamera([obj[i] for i in fit],
                                             [img[i] for i in fit], size, None, None,
                                             flags=flags)
    print(f"\ndistortion model '{args.dist_model}'")
    print(f"\nfit on {len(fit)} views, RMS reprojection {rms:.4f} px")

    held_err = None
    if len(held):
        errs = []
        for i in held:
            ok, rvec, tvec = cv2.solvePnP(obj[i], img[i], K, dist)
            if not ok:
                continue
            proj, _ = cv2.projectPoints(obj[i], rvec, tvec, K, dist)
            errs.append(float(np.sqrt(np.mean(
                np.sum((proj.reshape(-1, 2) - img[i].reshape(-1, 2)) ** 2, axis=1)))))
        held_err = float(np.mean(errs)) if errs else None
        print(f"held out {len(held)} views, reprojection {held_err:.4f} px"
              if held_err is not None else "held-out views did not solve")

    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    focal_mm = fx * DahengSource.PIXEL_SIZE_M * 1e3
    scales = [board_scale(c, pattern) for c in accepted]
    print(f"\n  fx {fx:9.2f}   fy {fy:9.2f}   (fy/fx {fy / fx:.5f})")
    print(f"  cx {cx:9.2f}   cy {cy:9.2f}   (image centre "
          f"{size[0] / 2:.1f}, {size[1] / 2:.1f})")
    print(f"  dist {np.array2string(dist.ravel(), precision=5, suppress_small=True)}")
    print(f"\n  implied focal length {focal_mm:.2f} mm on a "
          f"{DahengSource.PIXEL_SIZE_M * 1e6:.2f} um pixel")
    print(f"  coverage {coverage(accepted, shape) * 100:.0f} % of an 8x8 image grid")
    print(f"  board scale {min(scales):.0f}-{max(scales):.0f} px "
          f"(ratio {max(scales) / min(scales):.2f})")

    warnings = []
    if coverage(accepted, shape) < 0.5:
        warnings.append("coverage below 50 %: the distortion terms are unconstrained "
                        "where no corner ever landed, which is exactly the image edge "
                        "they matter most at")
    if max(scales) / min(scales) < 1.3:
        warnings.append("every view is at nearly the same distance: fx and the board "
                        "distance are poorly separated, so fx may be confident and wrong")
    if held_err is not None and held_err > 3 * rms:
        warnings.append(f"held-out error {held_err:.3f} px is over three times the fit "
                        f"error {rms:.3f}: the model is fitting these particular views")
    if abs(fy / fx - 1.0) > 0.01:
        warnings.append(f"fy/fx is {fy / fx:.4f}; square pixels are expected on this "
                        "sensor, so this suggests a thin or degenerate view set")
    for line in warnings:
        print(f"\n  !! {line}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "serial": args.serial,
        "width": size[0], "height": size[1],
        "fx": fx, "fy": fy, "cx": cx, "cy": cy,
        "dist": dist.ravel().tolist(),
        "square_mm": args.square_mm,
        "pattern": list(pattern),
        "views": len(accepted),
        "rms_fit_px": rms,
        "rms_heldout_px": held_err,
        "coverage": coverage(accepted, shape),
        "scale_ratio": max(scales) / min(scales),
        "implied_focal_mm": focal_mm,
        "source": "checkerboard",
        "dist_model": args.dist_model,
        "written": time.strftime("%Y-%m-%d %H:%M:%S"),
    }, indent=2) + "\n")
    print(f"\nwrote {args.out}")
    print("Load it into DahengSource as `intrinsics=` with `dist` -- "
          "`intrinsics_source` then reads 'calibrated' rather than 'nominal'.")
    return 0


def main() -> int:
    from dexrobot021_mjlab.deploy.camera import DahengSource

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--square-mm", type=float, required=True,
                    help="MEASURED checkerboard square, in mm. A wrong value "
                         "scales fx and every recovered distance by the same ratio "
                         "and the fit cannot see it.")
    ap.add_argument("--pattern", default=None, metavar="COLSxROWS",
                    help="inner corners; auto-detected from the first view if omitted")
    ap.add_argument("--serial", default=None, help="default: the first camera found")
    ap.add_argument("--views", type=int, default=20)
    ap.add_argument("--min-move-px", type=float, default=120.0,
                    help="a view must differ from every accepted one by at least "
                         "this mean corner displacement")
    ap.add_argument("--exposure-us", type=float, default=10000.0)
    ap.add_argument("--gain", type=float, default=9.0)
    ap.add_argument("--holdout", type=float, default=0.3)
    ap.add_argument("--out", type=Path, default=calib_dir() / "daheng_intrinsics.json")
    ap.add_argument("--timeout-s", type=float, default=300.0)
    ap.add_argument("--dist-model", default="k1k2",
                    choices=["k1", "k1k2", "k1k2p", "full"],
                    help="radial/tangential terms to fit. Default k1k2: on the 8 mm "
                         "lens all models predicted held-out corners equally well, but "
                         "the higher-order ones pulled the principal point ~60 px off "
                         "centre -- absorbing a bias, not describing the glass.")
    ap.add_argument("--from-npz", type=Path, default=None,
                    help="refit from corners saved by an earlier run instead of "
                         "collecting again -- the collection is the expensive part")
    args = ap.parse_args()

    pattern = None
    if args.pattern:
        cols, rows = (int(v) for v in args.pattern.lower().split("x"))
        pattern = (cols, rows)

    if args.from_npz is not None:
        saved = np.load(args.from_npz)
        accepted = [np.asarray(c, np.float64) for c in saved["corners"]]
        pattern = tuple(int(v) for v in saved["pattern"])
        shape = (int(saved["height"]), int(saved["width"]))
        args.square_mm = float(saved["square_mm"])
        print(f"refitting {len(accepted)} saved views from {args.from_npz}, "
              f"pattern {pattern[0]}x{pattern[1]}, square {args.square_mm:g} mm")
        return fit_and_report(accepted, pattern, shape, args)

    source = DahengSource(serial=args.serial, exposure_us=args.exposure_us,
                          gain=args.gain, nominal_focal_mm=12.0)
    source.start()
    shape = (source.intrinsics.height, source.intrinsics.width)
    print(f"camera {args.serial or '(first found)'}, {shape[1]}x{shape[0]}, "
          f"exposure {args.exposure_us:g} us, gain {args.gain:g} dB")
    print(f"square {args.square_mm:g} mm (measured), collecting {args.views} views\n")
    print("Move the board between views: change its distance AND its tilt, and "
          "work it into\nthe image corners. Views that look alike add nothing and "
          "the coverage figure below\nwill say so.\n")

    accepted: list[np.ndarray] = []
    deadline = time.time() + args.timeout_s
    try:
        while len(accepted) < args.views and time.time() < deadline:
            frame = source.read()
            if frame is None:
                continue
            if pattern is None:
                pattern = autodetect_pattern(frame.image)
                if pattern is None:
                    continue
                print(f"pattern locked to {pattern[0]}x{pattern[1]} inner corners "
                      f"({pattern[0] * pattern[1]} points)\n")
            corners = detect(frame.image, pattern)
            if corners is None:
                continue
            gap = novelty(corners, accepted, shape[1])
            if gap < args.min_move_px:
                continue
            accepted.append(corners)
            print(f"  view {len(accepted):2d}/{args.views}  "
                  f"moved {gap:6.0f} px  scale {board_scale(corners, pattern):6.1f} px"
                  f"  coverage {coverage(accepted, shape) * 100:4.0f} %")
            deadline = time.time() + args.timeout_s
    finally:
        source.stop()

    return fit_and_report(accepted, pattern, shape, args)


if __name__ == "__main__":
    raise SystemExit(main())
