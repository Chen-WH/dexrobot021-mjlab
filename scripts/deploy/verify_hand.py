"""Identify each joint's command-to-feedback map, one joint at a time.

    python scripts/deploy/verify_hand.py checkpoints/spin_cube          # prompts
    python scripts/deploy/verify_hand.py checkpoints/spin_cube --auto   # no prompts
    python scripts/deploy/verify_hand.py checkpoints/spin_cube --mock   # no hardware

**The hand must be empty.**  An object resting in it blocks the fingers, and a
blocked finger is indistinguishable from a badly scaled one.

`deploy/hand.py:JOINT_MAP` pairs each simulator joint with an SDK keyword.  The
pairing follows from the MJCF body tree and is reliable.  What it cannot know is
the **sign and zero of each axis**, which come from how the encoders are wired.
A flipped sign does not error; that finger just drives backwards, and the whole
policy looks like it failed to transfer.

Rather than sweep each joint and ask an operator "did that look right?", this
commands three setpoints per joint and fits the settled feedback against them:

    measured = slope * commanded + offset

which turns the question into three numbers with unambiguous readings:

    slope ~ +1, offset ~ 0     the mapping is right
    slope ~ -1                 the sign is flipped -> set `sign=-1.0` in JOINT_MAP
    |slope| << 1               the joint did not move: blocked, or not driven
    offset large               zero-point mismatch -> set `offset_deg` in JOINT_MAP

Direction and zero point are different claims and this measures both, which
asking a human cannot do -- an operator watching a finger flex cannot see a
five-degree zero offset.  It also removes the judgement call, so the record it
writes says what was measured rather than what somebody thought they saw.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "src"))

from dexrobot021_mjlab.deploy.contract import PolicyContract  # noqa: E402
from dexrobot021_mjlab.deploy.hand import JOINT_MAP, MockHand, PyzlgHand  # noqa: E402
from dexrobot021_mjlab.paths import calib_dir  # noqa: E402

#: Setpoints as a fraction of the distance from the default pose to the upper
#: limit. Deliberately short of the limit: if a sign is inverted the joint drives
#: the other way, and this keeps that from ending against a hard stop.
SETPOINT_FRACTIONS = (0.0, 0.30, 0.60)
RAMP_S = 0.5      # time to slew between setpoints
SETTLE_S = 0.6    # time to sit still before the feedback is trusted
SAMPLES = 12      # feedback reads averaged at each setpoint


def sweep_joint(hand, contract, index: int, dt: float) -> dict:
  """Drive one joint through the setpoints; return the fit."""
  base = np.asarray(contract.default_joint_pos, float)
  hi = np.asarray(contract.joint_limits, float)[index, 1]
  commanded, measured = [], []
  nan_before = int(getattr(hand, "nan_reads", np.zeros(1))[index]) \
    if hasattr(hand, "nan_reads") else 0
  reads_total = 0

  for fraction in SETPOINT_FRACTIONS:
    target = base.copy()
    target[index] = base[index] + fraction * (hi - base[index])

    start = hand.read()[0].copy()
    steps = max(int(RAMP_S / dt), 1)
    for k in range(steps):
      blend = (k + 1) / steps
      hand.send(start * (1.0 - blend) + target * blend)
      time.sleep(dt)
    deadline = time.time() + SETTLE_S
    while time.time() < deadline:
      hand.send(target)
      time.sleep(dt)

    reads = []
    for _ in range(SAMPLES):
      hand.send(target)
      reads.append(hand.read()[0][index])
      reads_total += 1
      time.sleep(dt)
    commanded.append(target[index])
    measured.append(float(np.mean(reads)))

  # Return to the default pose before the next joint.
  start = hand.read()[0].copy()
  steps = max(int(RAMP_S / dt), 1)
  for k in range(steps):
    blend = (k + 1) / steps
    hand.send(start * (1.0 - blend) + base * blend)
    time.sleep(dt)

  # How much of this joint's feedback was actually measured rather than held over
  # from a previous read. `PyzlgHand` holds the last good value through a NaN,
  # which is right for a blink and dangerous here: a joint whose feedback dropped
  # out reads as a perfectly constant angle, which looks exactly like a
  # mechanically blocked joint. Distinguishing them is the whole point.
  nan_after = int(getattr(hand, "nan_reads", np.zeros(1))[index]) \
    if hasattr(hand, "nan_reads") else 0
  held = (nan_after - nan_before) / max(reads_total, 1)

  c = np.asarray(commanded)
  m = np.asarray(measured)
  span = float(c.max() - c.min())
  if span < 1e-6 or not np.isfinite(m).all():
    return {"slope": float("nan"), "offset_deg": float("nan"),
            "residual_deg": float("nan"), "span_deg": np.degrees(span),
            "held_frac": round(held, 3), "max_error_deg": float("nan"),
            "commanded_deg": np.degrees(c).round(2).tolist(),
            "measured_deg": np.degrees(m).round(2).tolist()}
  slope, offset = np.polyfit(c, m, 1)
  residual = float(np.abs(m - (slope * c + offset)).max())
  # The error that matters is over the range the joint is used in, not the fit's
  # extrapolated intercept: a mild scale error produces a large intercept while
  # tracking well everywhere the joint is driven.
  max_error = float(np.abs(m - c).max())
  return {
    "slope": round(float(slope), 4),
    "offset_deg": round(float(np.degrees(offset)), 2),
    "residual_deg": round(float(np.degrees(residual)), 2),
    "span_deg": round(float(np.degrees(span)), 1),
    "held_frac": round(held, 3),
    "max_error_deg": round(float(np.degrees(max_error)), 2),
    "commanded_deg": np.degrees(c).round(2).tolist(),
    "measured_deg": np.degrees(m).round(2).tolist(),
  }


def verdict(fit: dict) -> tuple[str, str]:
  slope = fit["slope"]
  held = fit.get("held_frac", 0.0)
  if held > 0.5:
    # Checked before anything else: without it a dropped-out joint reads as a
    # constant angle and gets diagnosed as mechanically blocked.
    return "NO FEEDBACK", (
      f"{held * 100:.0f}% of reads came back NaN and were held at the last good "
      "value.\n    The angle below is that held value, not a measurement. The "
      "dropout appears\n    when a joint is driven into a stall (its board latches) "
      "-- check for an\n    obstruction before suspecting the CAN bus.")
  if not np.isfinite(slope):
    return "NO DATA", "feedback was NaN or the joint never moved"
  if slope < -0.5:
    return "SIGN FLIPPED", "set sign=-1.0 for this joint in JOINT_MAP"
  if abs(slope) < 0.5:
    return "NOT MOVING", "blocked (is the hand empty?), or the motor is not driven"
  error = fit.get("max_error_deg", 0.0)
  # Thresholds against the training-time observation noise (~0.19 deg on a 75 deg
  # joint). A static bias is more expensive than zero-mean noise; training
  # randomises encoder bias up to 3.4 deg.
  if error >= 5.0:
    return "TRACKING", (f"off by up to {error:.1f} deg in range "
                        f"(slope {slope:.3f}); check hall_scale for this joint")
  if error >= 2.0:
    return "bias", (f"off by up to {error:.1f} deg in range (slope {slope:.3f}) -- "
                    "usable, but the largest joint-level bias on the hand")
  return "ok", ""


def main() -> int:
  ap = argparse.ArgumentParser(description=__doc__,
                               formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("bundle", type=pathlib.Path)
  ap.add_argument("--mock", action="store_true")
  ap.add_argument("--auto", action="store_true",
                  help="no prompts. Only with an operator watching and an e-stop "
                       "in reach: this moves every finger.")
  ap.add_argument("--can-device", type=int, default=0)
  ap.add_argument("--only", default=None, help="one joint, by simulator name")
  ap.add_argument("--record", default=None, metavar="METHOD",
                  help="record the mapping as confirmed by other means, without "
                       "moving anything. The method is stored alongside, because "
                       "checking axis directions on paper and measuring the "
                       "command-to-feedback map are different claims.")
  args = ap.parse_args()

  contract = PolicyContract.load(args.bundle / "contract.json")
  out = calib_dir() / "joint_verification.json"
  out.parent.mkdir(parents=True, exist_ok=True)

  if args.record:
    out.write_text(json.dumps({
      **{n: "ok" for n in contract.joint_names},
      "_method": args.record,
      "_moved_the_hand": False,
    }, indent=2) + "\n")
    print(f"recorded {len(contract.joint_names)} joints as confirmed by:\n"
          f"  {args.record}\nwrote {out}")
    return 0

  dt = contract.control_dt
  hand = (MockHand(contract.joint_names, np.asarray(contract.default_joint_pos), dt)
          if args.mock
          else PyzlgHand(contract.joint_names, device_index=args.can_device))

  print(f"{len(contract.joint_names)} joints, {len(SETPOINT_FRACTIONS)} setpoints "
        f"each, ~{(len(SETPOINT_FRACTIONS) + 1) * (RAMP_S + SETTLE_S):.0f} s per joint")
  print("THE HAND MUST BE EMPTY -- an object in it blocks the fingers and reads as a "
        "scale error.\n")
  if not args.auto and not args.mock:
    try:
      input("press Enter to begin, Ctrl-C to abort: ")
    except EOFError:
      print("\nno terminal: re-run directly, or pass --auto if an operator is "
            "watching with an e-stop in reach.")
      return 1

  hand.connect()
  results: dict[str, object] = {}
  try:
    for i, name in enumerate(contract.joint_names):
      if args.only and name != args.only:
        continue
      sdk = {m.sim_name: m.sdk_name for m in JOINT_MAP}[name]
      print(f"[{i + 1:2d}/{len(contract.joint_names)}] {name:14s} -> {sdk:7s} ",
            end="", flush=True)
      fit = sweep_joint(hand, contract, i, dt)
      tag, hint = verdict(fit)
      results[name] = {**fit, "verdict": tag}
      print(f"slope {fit['slope']:+.3f}  max err {fit['max_error_deg']:5.2f} deg  "
            f"held {fit['held_frac'] * 100:3.0f}%  [{tag}]"
            + (f"\n    {hint}" if hint else ""))
  except KeyboardInterrupt:
    print("\ninterrupted")
  finally:
    hand.close()

  bad = [n for n, v in results.items() if v["verdict"] not in ("ok", "bias")]
  noted = [n for n, v in results.items() if v["verdict"] == "bias"]
  if args.mock:
    # A mock run must never certify the hardware: `MockHand` follows every command
    # exactly, so it would report twelve perfect joints over a real record.
    print("\n--mock: results NOT recorded (a simulated hand cannot verify a real one)")
    return 0 if not bad else 1

  payload = {
    **{n: ("ok" if v["verdict"] in ("ok", "bias") else v["verdict"])
       for n, v in results.items()},
    "_method": (f"measured command-to-feedback fit over {len(SETPOINT_FRACTIONS)} "
                f"setpoints per joint"),
    "_moved_the_hand": True,
    "_complete": len(results) == len(contract.joint_names),
    "_fits": results,
  }
  out.write_text(json.dumps(payload, indent=2) + "\n")
  print(f"\nwrote {out}")

  if bad:
    print(f"\n{len(bad)} joint(s) need attention: {', '.join(bad)}")
    return 1
  if noted:
    print(f"\n{len(noted)} joint(s) usable but biased: {', '.join(noted)}")
  if len(results) == len(contract.joint_names):
    worst = max(v["max_error_deg"] for v in results.values())
    print(f"\nAll 12 joints track their commands; worst in-range error "
          f"{worst:.2f} deg.")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
