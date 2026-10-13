# Copyright 2026 Luc Trudeau
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Adaptive scheduling for joint perceptual optimization."""

import math
from dataclasses import dataclass


@dataclass
class AdaptiveLPIPSScheduler:
    """Select LPIPS networks from periodically measured distortions."""

    check_every: int = 10
    threshold: float = 0.02
    active: str = "both"

    def __post_init__(self) -> None:
        """Validate scheduling parameters."""
        if self.check_every < 1:
            raise ValueError("check_every must be positive.")

        if not math.isfinite(self.threshold) or not 0 <= self.threshold < 1:
            raise ValueError("threshold must be finite and in [0, 1).")

    def select(self, step: int) -> str:
        """Choose the metric or metrics to evaluate at this step."""
        if step < 0:
            raise ValueError("step must be nonnegative.")

        if step <= 1 or step % self.check_every == 0:
            return "both"

        return self.active

    def update(self, alex: float, vgg: float) -> None:
        """Update the active metric using normalized distortions."""
        if not all(math.isfinite(value) and value >= 0 for value in (alex, vgg)):
            raise ValueError("Normalized distortions must be finite and nonnegative.")

        gap = abs(alex - vgg) / max(alex, vgg, 1e-8)

        if gap <= self.threshold:
            self.active = "both"
        else:
            self.active = "alex" if alex > vgg else "vgg"
