# Third-party notices

This release retains research components adapted through the following projects.
The accompanying upstream license texts apply to those components; a LeaP
top-level license does not replace third-party terms.

| Project | Components in this release | Upstream license |
| --- | --- | --- |
| [Diffusion Policy](https://github.com/real-stanford/diffusion_policy) | Normalizer, tensor dictionary/device utilities and base policy ancestry | MIT, Copyright (c) 2023 Columbia Artificial Intelligence and Robotics Lab; [text](licenses/diffusion-policy-MIT.txt) |
| [3D Diffusion Policy](https://github.com/YanjieZe/3D-Diffusion-Policy) | PointNet encoders and robot-policy utilities, via FlowPolicy | MIT, Copyright (c) 2024 Yanjie Ze; [text](licenses/dp3-MIT.txt) |
| [FlowPolicy](https://github.com/zql-kk/FlowPolicy) | Research policy scaffold, utilities and positional embedding ancestry | MIT, Copyright (c) 2025 zql-kk; [text](licenses/flowpolicy-MIT.txt) |
| [A2A / RoboVerse](https://github.com/JIAjindou/A2A_Flow_Matching) | SimpleFlowNet architecture and flow-matching implementation adapted through the research A2A port | Apache-2.0; [text](licenses/a2a-Apache-2.0.txt) |

LeaP modifications include packaging these components independently, adding the
state-conditioned source prior, composing the three losses, and integrating
RoboTwin joint-space training and deployment. The flow network uses the research
initialization behavior and the matcher accepts an externally sampled source.

[RoboTwin](https://github.com/RoboTwin-Platform/RoboTwin) is an external pinned
checkout, not vendored code. Its original MIT license and copyright notice remain
in that checkout. Downloaded assets and installed dependencies retain their own
terms. No simulator assets, demonstration datasets or model weights are included.
