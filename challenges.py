"""Challenge-response system for active liveness verification."""

import random
import time
from dataclasses import dataclass, field
from enum import Enum, auto


class ChallengeType(Enum):
    BLINK      = auto()
    TURN_LEFT  = auto()
    TURN_RIGHT = auto()
    THUMBS_UP  = auto()
    SHOW_PINKY = auto()


_INSTRUCTIONS: dict[ChallengeType, str] = {
    ChallengeType.BLINK:      "Blink your eyes",
    ChallengeType.TURN_LEFT:  "Turn your head to the LEFT",
    ChallengeType.TURN_RIGHT: "Turn your head to the RIGHT",
    ChallengeType.THUMBS_UP:  "Show a THUMBS UP",
    ChallengeType.SHOW_PINKY: "Raise your LITTLE FINGER",
}

# Frames the condition must hold continuously before the challenge is accepted.
# Blink uses edge-triggered detection (EAR low → high) so sustain is 0.
_SUSTAIN_FRAMES: dict[ChallengeType, int] = {
    ChallengeType.BLINK:      0,
    ChallengeType.TURN_LEFT:  5,
    ChallengeType.TURN_RIGHT: 5,
    ChallengeType.THUMBS_UP:  5,
    ChallengeType.SHOW_PINKY: 5,
}

HAND_CHALLENGES = {ChallengeType.THUMBS_UP, ChallengeType.SHOW_PINKY}


@dataclass
class Challenge:
    ctype: ChallengeType
    timeout: float = 6.0
    _start: float = field(default_factory=time.time, init=False, repr=False)

    @property
    def instruction(self) -> str:
        return _INSTRUCTIONS[self.ctype]

    @property
    def time_remaining(self) -> float:
        return max(0.0, self.timeout - (time.time() - self._start))

    @property
    def is_expired(self) -> bool:
        return time.time() - self._start > self.timeout


class ChallengeRunner:
    def __init__(self, count: int = 2, types: "list[ChallengeType] | None" = None):
        if types is not None:
            chosen = types
        else:
            chosen = random.sample(list(ChallengeType), min(count, len(ChallengeType)))
        self.challenges: list[Challenge] = [Challenge(t) for t in chosen]
        self.index: int = 0
        self._blink_frames   = 0   # consecutive frames with EAR below threshold
        self._sustain_frames = 0   # consecutive frames satisfying a non-blink condition

    @property
    def current(self) -> "Challenge | None":
        return self.challenges[self.index] if self.index < len(self.challenges) else None

    @property
    def all_done(self) -> bool:
        return self.index >= len(self.challenges)

    def update(self, face_data, hand_data=None) -> bool:
        """Evaluate the current challenge. Returns True if just completed."""
        c = self.current
        if c is None:
            return False

        if c.ctype == ChallengeType.BLINK:
            # Edge-triggered: detect EAR low → high transition
            if face_data.is_blinking:
                self._blink_frames += 1
            else:
                if self._blink_frames >= 2:
                    self._advance()
                    return True
                self._blink_frames = 0

        else:
            # Sustained-detection challenges
            if c.ctype == ChallengeType.TURN_LEFT:
                active = face_data.is_turned_left
            elif c.ctype == ChallengeType.TURN_RIGHT:
                active = face_data.is_turned_right
            elif c.ctype == ChallengeType.THUMBS_UP:
                active = hand_data is not None and hand_data.is_thumbs_up
            elif c.ctype == ChallengeType.SHOW_PINKY:
                active = hand_data is not None and hand_data.is_pinky_raised
            else:
                active = False

            if active:
                self._sustain_frames += 1
                if self._sustain_frames >= _SUSTAIN_FRAMES[c.ctype]:
                    self._advance()
                    return True
            else:
                self._sustain_frames = 0

        return False

    def _advance(self):
        self.index          += 1
        self._blink_frames   = 0
        self._sustain_frames = 0
        # Reset the timer on the newly active challenge so each gets a full window
        if self.index < len(self.challenges):
            self.challenges[self.index]._start = time.time()
