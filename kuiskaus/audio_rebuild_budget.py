"""The rebuild budget behind the device-move reset (issue #72).

``reset_session`` rebuilds the session when the live default moved, and a
device that never settles must not cost a full terminate-all +
``Pa_Initialize`` + re-enumeration on every press -- that stall lands on the
hotkey callback thread. Two counters see off "this is a flapping device, not
a transition", because the flapping is not one shape:

* one device whose ``isAlive`` flag oscillates through a coreaudiod storm --
  every read is a different pair, the dedupe guard is structurally blind to
  it, and the same-device ``streak`` is what says so (and names the device
  in the verdict);
* a default alternating between two devices (the AirPod drop/rejoin, the
  issue's own scenario) -- each press legitimately sees a DIFFERENT device,
  so any per-device count restarts on every press and never fires. The
  bound for that shape is ``count``: consecutive rebuilds whatever device
  they name, since the session last agreed with the live device.

A device the session has not rebuilt against during this run is progress --
a user moving the default by hand, the condition issue #72 exists to
follow -- so the counters restart for it; a device it HAS rebuilt against is
the counted pair bouncing back, and counting continues.
"""

from __future__ import annotations

# Consecutive rebuilds one session may be asked for before it says the
# device is not settling. The dedupe guard stops a STABLE pair from costing
# a rebuild per press, but it cannot help a pair CoreAudio reports
# differently on every read: each press then legitimately needs the reset,
# and each press pays a full teardown on the hotkey callback thread. Three
# is "this is not a transition, it is a flapping device"; the reset's
# backoff still retries, so a device that settles is recovered without an
# app restart.
MAX_RESET_REBUILDS = 3


class RebuildBudget:
    """The consecutive rebuilds that bound the reset's churn.

    Owned by ``PyAudioRegistry`` (one per recorder session), consumed by
    ``audio_device_reset`` -- the module that decides when a rebuild has
    actually run. Plain attributes, no lock: every write comes from the
    thread running a reset, and a lost or stale read here costs one extra
    rebuild or one delayed verdict, never a terminated PortAudio under a
    live thread (the ledgers gate that, not this).
    """

    def __init__(self) -> None:
        # Consecutive rebuilds against the SAME device -- the oscillating-
        # liveness shape. A device change restarts it, because a change of
        # device is progress and a user switching microphones by hand must
        # never be told the device "never settled" (the message names it).
        self.streak = 0
        self.device: int | None = None
        # Consecutive rebuilds since the session last agreed with the live
        # device, against ANY device -- the alternating-default shape the
        # streak above is structurally blind to.
        self.count = 0
        # The devices rebuilt against during this run. "Already counted":
        # the backoff's new-evidence exemption honours a device only until
        # it has been rebuilt against once, so a pair alternating A<->B
        # cannot keep the exemption open forever.
        self.recent: list[int] = []

    def note_rebuild(self, device_id: int) -> None:
        """Record one completed rebuild against ``device_id``."""
        self.streak = self.streak + 1 if self.device == device_id else 1
        self.device = device_id
        counted = device_id in self.recent
        self.count = self.count + 1 if counted else 1
        if not counted:
            self.recent.append(device_id)

    def settle(self) -> None:
        """A press found the session bound to what the system reports.

        The churn stopped, so every view of the run restarts from zero --
        the next storm pays for itself from scratch.
        """
        self.streak = 0
        self.device = None
        self.count = 0
        self.recent = []

    def is_new_device(self, device_id: int) -> bool:
        """Whether the session has never rebuilt against ``device_id``
        during this run.

        That is what a hand-moved default looks like, and it is the evidence
        a refused teardown is waiting for. A device already in ``recent`` is
        the counted pair bouncing back: honouring it would keep the
        exemption permanently open during a two-device flap and spend the
        full futile teardown on every press.
        """
        return device_id not in self.recent
