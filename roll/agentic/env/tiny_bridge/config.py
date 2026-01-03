from dataclasses import dataclass, field
from typing import Tuple, Optional, Dict, List


@dataclass
class TinyBridgeConfig:
    seed: int = 42
    render_mode: str = "text"
    opponent_difficulty: str = "random" 

