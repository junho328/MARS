from dataclasses import dataclass, field
from typing import Tuple, Optional, Dict, List


@dataclass
class BridgeConfig:
    seed: int = 42
    render_mode: str = "text"
    built_in_opponent: str = "random"  # Options: "none", "random"
    opponent_player: int = 1
    include_opponent_turn: str = "action"

    # Contract Bridge specific config
    dealer: int = 0  # 0=North, 1=East, 2=South, 3=West
    dealer_vul: bool = False  # Dealer side vulnerability
    non_dealer_vul: bool = False  # Non-dealer side vulnerability
    use_double_dummy_result: bool = True  # Use double-dummy result for scoring
