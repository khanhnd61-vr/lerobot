# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Client for a `vla.cpp` inference server.

`vla-server` answers ZeroMQ REQ/REP requests carrying protobuf, which is a different
wire format from lerobot's own gRPC async-inference protocol. `vla_pb2.py` is
generated from the vendored `vla.proto`; regenerate both together if the engine's
schema changes:

```bash
python -m grpc_tools.protoc -I src/lerobot/vla_cpp \
    --python_out=src/lerobot/vla_cpp vla.proto
```
"""

from .archs import ARCH_PRESETS
from .async_client import AGGREGATE_FUNCTIONS, AsyncVlaCppClient, TimedActionQueue
from .client import DEFAULT_ADDRESS, VlaCppClient, VlaCppError
from .stats import build_gr00t_normalizers, pi05_state_quantiles

__all__ = [
    "AGGREGATE_FUNCTIONS",
    "ARCH_PRESETS",
    "AsyncVlaCppClient",
    "DEFAULT_ADDRESS",
    "TimedActionQueue",
    "VlaCppClient",
    "VlaCppError",
    "build_gr00t_normalizers",
    "pi05_state_quantiles",
]
