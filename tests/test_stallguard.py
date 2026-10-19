"""Stall guard against a toy plant with the three behaviours measured on the hand:
deadband ~0.5 deg, rate limit, and a latch when the joint is stalled with the target > 0.6 deg
ahead for >= 0.2 s. Readings are quantised to 0.55 deg and toggle when resting on a boundary.

    python -m pytest tests/test_stallguard.py
"""
import contextlib

import numpy as np

from dexrobot021_mjlab.deploy.stallguard import StallGuardHand

DT, D = 0.016, np.radians


class Plant:
  def __init__(self, block=None, block_until=None):
    self.q = 0.0; self.cmd = 0.0; self.block = block; self.block_until = block_until
    self.k = 0; self.stall_t = 0.0; self.latched = False; self.latch_step = None
  def read(self):
    # quantised to 0.55 deg; a joint resting between two levels toggles between them
    lo = np.floor(self.q / D(0.55)) * D(0.55)
    frac = (self.q - lo) / D(0.55)
    level = lo + D(0.55) * ((self.k % 2) if 0.2 < frac < 0.8 else round(frac))
    return np.array([level]), np.zeros(1)
  def send(self, t):
    self.cmd = float(t[0]); self.k += 1
    if self.latched: return
    err = self.cmd - self.q
    blocked_now = self.block is not None and (self.block_until is None or self.k < self.block_until)
    step = np.clip(err, -D(300) * DT, D(300) * DT) if abs(err) > D(0.5) else 0.0
    new = self.q + step
    if blocked_now and new > self.block: new = self.block
    stalled = blocked_now and self.q >= self.block - 1e-9 and err > D(0.6)
    self.stall_t = self.stall_t + DT if stalled else 0.0
    if self.stall_t >= 0.2: self.latched = True; self.latch_step = self.k
    self.q = new


def run(plant, targets, guard=True):
  # The guard is only active inside `during_wait()` (the operator place/remove holds).
  h = StallGuardHand(plant, 1) if guard else plant
  with h.during_wait() if guard else contextlib.nullcontext():
    for t in targets:
      h.read(); h.send(np.array([t]))
  return h


def test_free_steps_from_rest_are_not_frozen():
  p = Plant()
  tg = np.concatenate([np.full(40, D(0.0)), np.full(80, D(5.0)), np.full(80, D(12.0)), np.full(80, D(11.0))])
  run(p, tg)
  assert abs(p.q - D(11.0)) < D(0.6), np.degrees(p.q)


def test_blocked_joint_never_latches_with_guard_and_does_without():
  tg = np.concatenate([np.full(20, 0.0), np.full(400, D(30.0))])
  bare = Plant(block=D(10.27)); run(bare, tg, guard=False)
  assert bare.latched
  p = Plant(block=D(10.27)); h = run(p, tg)
  assert not p.latched
  assert h.clamp_events[0] >= 1 and h.clamped_steps[0] > 200


def test_release_when_obstacle_goes_away():
  tg = np.concatenate([np.full(20, 0.0), np.full(600, D(30.0))])
  p = Plant(block=D(10.27), block_until=300); run(p, tg)
  assert not p.latched
  assert p.q > D(29.0), np.degrees(p.q)


def test_holding_joint_pushed_off_target_is_clamped():
  # target fixed, an external load displaces the joint 1 deg and holds it there
  class Pushed(Plant):
    def send(self, t):
      super().send(t); self.q = min(self.q, D(19.0)) if self.k > 60 else self.q
  p = Pushed(block=D(19.0)); tg = np.full(400, D(20.0))
  h = run(p, tg)
  assert not p.latched and h.clamp_events[0] >= 1



def test_outside_a_wait_the_policy_target_passes_through():
  # Episode (outside during_wait): identical to no guard -> the blocked joint latches.
  tg = np.concatenate([np.full(20, 0.0), np.full(400, D(30.0))])
  p = Plant(block=D(10.27)); h = StallGuardHand(p, 1)
  for t in tg:
    h.read(); h.send(np.array([t]))
  assert p.latched and h.clamped_steps[0] == 0
  # and after a wait the guard is off again with its state cleared
  h = StallGuardHand(Plant(block=D(10.27)), 1)
  with h.during_wait():
    for t in tg:
      h.read(); h.send(np.array([t]))
  assert h.clamped_steps[0] > 200 and not h.active and not h._clamped.any()

def test_a_wait_holds_its_command_still_while_a_person_pries_a_finger_open():
  """The operator pries a joint away from its target (about 1 deg per 128 ms, as on
  the rig) and then holds it there. Two things must be true at once:

  * the guard must not drive current into the pried joint -- it clamps, so the
    command stays within a degree or so of where the joint actually is once the
    detector calls it blocked;
  * the REQUESTED pose must still be the one the caller asked for when the wait
    ends, because `_hold_while` is what leaves the hand at a determinate place for
    the closing leg to start from. (A "follow the measured angle" mode satisfies
    the first and breaks the second.)

  While the finger is MOVING it is making progress, so it is not called blocked and
  the lead grows to several degrees; that is paid during a wait in which the board
  is not being asked to hold anything.
  """
  class Pried(Plant):
    def send(self, t):
      # the operator's hand IS the block, and it moves
      self.block = D(20.0) - D(min(10.0, 0.125 * max(0, self.k - 40)))
      super().send(t)
      self.q = min(self.q, self.block)

  p = Pried(block=D(20.0)); p.q = D(20.0)
  h = StallGuardHand(p, 1)
  cmds = []
  with h.during_wait():
    for _ in range(900):
      h.read(); h.send(np.array([D(20.0)]))
      cmds.append(p.cmd)
  assert not p.latched
  # once the operator stops, the command sits on the joint again ...
  assert abs(cmds[-1] - p.q) < D(0.6), np.degrees(abs(cmds[-1] - p.q))
  # ... and the REQUEST was never rewritten: the caller's pose is what it asked for.
  assert np.allclose(h.last_requested, D(20.0))


if __name__ == "__main__":
  for name, fn in list(globals().items()):
    if name.startswith("test_"):
      fn(); print("PASS", name)
