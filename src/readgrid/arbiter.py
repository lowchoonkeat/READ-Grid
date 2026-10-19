"""Authority arbiter and command fence (Phase 6).

One arbiter per edge domain. **No cloud-to-actuator bypass**: every actuation
command, whatever its source, is submitted to the domain arbiter and either
accepted or rejected.

Command fields
--------------
``command_id``, ``source``, ``authority_epoch``, ``sequence_number``,
``creation_time``, ``expiry_time``, ``requested_action``.

Acceptance requires *all* of:
* the source is the currently authorised source;
* the epoch equals the arbiter's current epoch;
* the sequence number is newer than the last accepted sequence from that source;
* the command has not expired at the moment of acceptance;
* the requested action is admissible (ramp / dwell / burden, enforced by the
  caller-supplied ``admissible`` predicate).

States
------
``CONNECTED -> SUSPECT -> LOCAL_FALLBACK -> RECOVERING -> CONNECTED``

Boundary semantics are ``>=``: SUSPECT at ``>= 3.0 s`` and LOCAL_FALLBACK at
``>= 5.0 s`` since the last *valid* heartbeat arrival. At equal timestamps the
watchdog expiry is processed **before** a packet arrival (documented tie-break,
see ``docs/audit_resolution.md`` A8), so authority transfer is the conservative
default.

Restart resilience is explicitly **excluded** from the tested scope (see
``docs/blockers.md`` B9); persistent epoch/sequence state across a restart is
not exercised.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field

CONNECTED = "CONNECTED"
SUSPECT = "SUSPECT"
LOCAL_FALLBACK = "LOCAL_FALLBACK"
RECOVERING = "RECOVERING"

ACCEPT = "accepted"
REJ_WRONG_SOURCE = "rejected:wrong_source"
REJ_EPOCH = "rejected:stale_epoch"
REJ_SEQ = "rejected:duplicate_or_out_of_order"
REJ_EXPIRED = "rejected:expired"
REJ_INADMISSIBLE = "rejected:inadmissible_action"

_CMD_ID = itertools.count(1)


@dataclass
class Command:
    source: str
    requested_action: dict     # {group_id: action level}
    authority_epoch: int
    sequence_number: int
    creation_time: float
    expiry_time: float
    domain_id: str
    command_id: int = field(default_factory=lambda: next(_CMD_ID))

    def is_expired(self, t: float) -> bool:
        return t >= self.expiry_time


@dataclass
class ArbiterEvent:
    t: float
    domain_id: str
    kind: str
    detail: str = ""


class DomainArbiter:
    def __init__(self, domain_id: str, protocol, initial_authority: str = "cloud"):
        self.domain_id = domain_id
        self.p = protocol
        self.state = CONNECTED
        self.authority = initial_authority
        self.epoch = 0

        self.last_hb_time = 0.0
        self.link_up_since: float | None = 0.0
        self.last_transfer_time = -1e9
        self.last_accepted_seq: dict[str, int] = {"cloud": -1, "local": -1}
        self.last_accepted_cmd: Command | None = None
        self.last_accepted_action: int = 0
        self.last_action_change_time: float = -1e9

        # counters
        self.n_transfers = 0
        self.n_accepted = 0
        self.n_rejected = 0
        self.reject_reasons: dict[str, int] = {}
        self.events: list[ArbiterEvent] = []
        # timestamped authority-epoch history (for transfer-delay metrics)
        self.transfer_times: list[tuple[float, str]] = []

    # ------------------------------------------------------------- heartbeat
    def on_heartbeat(self, t: float, source: str = "cloud") -> list[ArbiterEvent]:
        if source != "cloud":
            return []
        self.last_hb_time = t
        if self.link_up_since is None:
            self.link_up_since = t
        ev: list[ArbiterEvent] = []
        # stable-link guard: only return to cloud authority after the link has
        # been continuously valid for recovery_stable_s
        if self.state in (SUSPECT, LOCAL_FALLBACK, RECOVERING):
            if self.state != RECOVERING:
                self._set_state(RECOVERING, t, ev, "heartbeat resumed")
            if (t - self.link_up_since) >= self.p.raw["authority"]["recovery_stable_s"] \
                    and (t - self.last_transfer_time) >= self.p.raw["authority"]["min_dwell_s"]:
                self._transfer("cloud", t, ev, "stable link recovered")
        return ev

    def on_heartbeat_loss(self, t: float) -> None:
        """Mark the link as not currently valid (used by lossy models)."""
        self.link_up_since = None

    # ------------------------------------------------------------------ tick
    def on_tick(self, t: float) -> list[ArbiterEvent]:
        ev: list[ArbiterEvent] = []
        age = t - self.last_hb_time
        if self.authority == "cloud":
            if age >= self.p.raw["authority"]["takeover_after_s"]:
                if self.state != LOCAL_FALLBACK:
                    self._set_state(LOCAL_FALLBACK, t, ev, f"no heartbeat {age:.2f}s")
                if (t - self.last_transfer_time) >= self.p.raw["authority"]["min_dwell_s"]:
                    self._transfer("local", t, ev, "takeover")
            elif age >= self.p.raw["authority"]["suspect_after_s"]:
                if self.state != SUSPECT:
                    self._set_state(SUSPECT, t, ev, f"no heartbeat {age:.2f}s")
        return ev

    # -------------------------------------------------------------- transfer
    def _set_state(self, new: str, t: float, ev: list[ArbiterEvent], why: str) -> None:
        if new == self.state:
            return
        old = self.state
        self.state = new
        self.events.append(ArbiterEvent(t, self.domain_id, "state", f"{old}->{new} ({why})"))
        ev.append(self.events[-1])

    def _transfer(self, new_auth: str, t: float, ev: list[ArbiterEvent], why: str) -> None:
        if new_auth == self.authority:
            return
        old = self.authority
        self.authority = new_auth
        self.epoch += 1                      # atomic change + epoch increment
        self.last_transfer_time = t
        self.n_transfers += 1
        self.transfer_times.append((t, f"{old}->{new_auth}"))
        # A new authority starts a fresh sequence space for that source. Old
        # commands are fenced by the epoch check, which is the primary fence;
        # the per-source sequence space is reset so that the new authority's
        # numbering does not have to be monotone across epochs.
        self.last_accepted_seq[new_auth] = -1
        self.events.append(ArbiterEvent(t, self.domain_id, "authority",
                                        f"{old}->{new_auth} epoch={self.epoch} ({why})"))
        ev.append(self.events[-1])
        if new_auth == "cloud":
            self._set_state(CONNECTED, t, ev, "authority restored")

    # ------------------------------------------------------------- acceptance
    def accept(self, cmd: Command, t: float, *, admissible=None,
               strict: bool = True) -> tuple[bool, str]:
        """Accept or reject a command.

        ``strict=False`` reproduces the *unguarded, heartbeat-only* diagnostic
        variant used by the Phase 11 ablation: the epoch, sequence and expiry
        fences are bypassed and only the source check is kept.
        """
        if cmd.domain_id != self.domain_id:
            return self._reject(REJ_WRONG_SOURCE)
        if cmd.source != self.authority:
            return self._reject(REJ_WRONG_SOURCE)
        if strict:
            if cmd.authority_epoch != self.epoch:
                return self._reject(REJ_EPOCH)
            if cmd.sequence_number <= self.last_accepted_seq.get(cmd.source, -1):
                return self._reject(REJ_SEQ)
            if cmd.is_expired(t):
                return self._reject(REJ_EXPIRED)
        if admissible is not None and not admissible(cmd.requested_action, t):
            return self._reject(REJ_INADMISSIBLE)
        self.last_accepted_seq[cmd.source] = cmd.sequence_number
        self.last_accepted_cmd = cmd
        self.n_accepted += 1
        act = cmd.requested_action
        prev = self.last_accepted_action
        if isinstance(prev, dict) and isinstance(act, dict):
            changed = any(act.get(g, 0) != prev.get(g, 0) for g in act)
        else:
            changed = act != prev
        if changed:
            self.last_action_change_time = t
        self.last_accepted_action = dict(act) if isinstance(act, dict) else act
        return True, ACCEPT

    def _reject(self, reason: str) -> tuple[bool, str]:
        self.n_rejected += 1
        self.reject_reasons[reason] = self.reject_reasons.get(reason, 0) + 1
        return False, reason

    # ---------------------------------------------------------------- summary
    def summary(self) -> dict:
        return {
            "domain_id": self.domain_id,
            "final_state": self.state,
            "final_authority": self.authority,
            "final_epoch": self.epoch,
            "n_transfers": self.n_transfers,
            "n_accepted": self.n_accepted,
            "n_rejected": self.n_rejected,
            "reject_reasons": dict(self.reject_reasons),
            "transfer_times": self.transfer_times,
        }
