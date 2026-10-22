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

A device the session has not rebuilt against recently is progress -- a user
moving the default by hand, the condition issue #72 exists to follow -- and
the backoff's new-evidence exemption honours it; a device it HAS rebuilt
against is the counted pair bouncing back, and the exemption does not apply.
The exemption is the only use of novelty: the bound itself counts every
rebuild, because a stream of ids the session has never seen is the same
churn wearing a different name.
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

# How many counted device ids the novelty window remembers. A window, not a
# ledger: the exemption needs to tell "the counted pair came back" from "a
# device nobody has rebuilt against", and a two-device flap -- the shape the
# exemption exists to refuse -- keeps both ids inside eight.
RECENT_DEVICE_WINDOW = 8

# How many times the new-evidence exemption may lift the backoff since the
# session last settled. The window above bounds WHICH ids read as unseen;
# this bounds how often an unseen one buys the full futile teardown. Without
# it a default reported as a DIFFERENT id on every press, rotating through
# more ids than the window holds, evicts its own history: every press reads
# as "new evidence", the exemption never closes, and the measured cost was
# 35 full terminate-all + ``Pa_Initialize`` cycles over 40 presses. A
# hand-moved default does not pay for this: the press after it finds the
# session agreeing with the live device and ``settle()`` restores the
# allowance, and once the allowance is spent the hand-switch still lands
# when the backoff window expires.
MAX_NEW_EVIDENCE_EXEMPTIONS = 8


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
        # streak above is structurally blind to. Never restarted by an
        # unseen id: a default reported as a different device on every press
        # is the churn this bound exists to stop, not progress through it.
        self.count = 0
        # The last few device ids counted. The backoff's new-evidence
        # exemption honours a device only until it has been rebuilt against,
        # so a pair alternating A<->B cannot keep the exemption open forever.
        self.recent: list[int] = []
        # Exemptions claimed since the last settle -- see
        # MAX_NEW_EVIDENCE_EXEMPTIONS for why the exemption needs a bound of
        # its own next to the window.
        self.new_evidence_spent = 0

    def note_rebuild(self, device_id: int) -> None:
        """Record one completed rebuild against ``device_id``."""
        self.streak = self.streak + 1 if self.device == device_id else 1
        self.device = device_id
        self.count += 1
        if device_id not in self.recent:
            self.recent.append(device_id)
            if len(self.recent) > RECENT_DEVICE_WINDOW:
                del self.recent[:-RECENT_DEVICE_WINDOW]

    def settle(self) -> None:
        """A press found the session bound to what the system reports.

        The churn stopped, so every view of the run restarts from zero --
        the next storm pays for itself from scratch.
        """
        self.streak = 0
        self.device = None
        self.count = 0
        self.recent = []
        self.new_evidence_spent = 0

    def is_new_device(self, device_id: int) -> bool:
        """Whether the session has not rebuilt against ``device_id``
        recently.

        That is what a hand-moved default looks like, and it is the evidence
        a refused teardown is waiting for. A device inside ``recent`` is the
        counted pair bouncing back: honouring it would keep the exemption
        permanently open during a two-device flap and spend the full futile
        teardown on every press.
        """
        return device_id not in self.recent

    def claim_exemption(self) -> bool:
        """Claim one new-evidence exemption (paired with ``is_new_device``).

        False once ``MAX_NEW_EVIDENCE_EXEMPTIONS`` have been spent since the
        last settle: an id rotation faster than the window would otherwise
        keep the exemption permanently open (each rotation evicts its own
        history, so the window's novelty check fires on every press), and the
        backoff the exemption suspends would never hold.
        """
        if self.new_evidence_spent >= MAX_NEW_EVIDENCE_EXEMPTIONS:
            return False
        self.new_evidence_spent += 1
        return True
