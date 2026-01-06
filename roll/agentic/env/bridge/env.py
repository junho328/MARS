import matplotlib.pyplot as plt
import numpy as np
import os
import random
from roll.agentic.env.bridge.config import BridgeConfig
from roll.agentic.utils import all_seed
from roll.agentic.env.base import BaseDiscreteActionEnv
from textwrap import dedent
import json
from typing import Optional, Dict, Any, List
import re
from PIL import Image
from collections import deque
import pyspiel
import warnings

class Bridge(BaseDiscreteActionEnv):
    """Contract Bridge game environment using OpenSpiel.
    
    Full Contract Bridge with 4 players, 52 cards, and 90 actions.
    Modified to handle Phase detection (Bidding vs Play) and Dummy visibility.
    """

    # Player names for Contract Bridge
    PLAYER_NAMES = {
        0: 'North',
        1: 'East',
        2: 'South',
        3: 'West',
    }

    def __init__(self, config: BridgeConfig = BridgeConfig()):
        self.config = config
        self.render_mode = config.render_mode
        self.built_in_opponent = config.built_in_opponent
        self.opponent_player = config.opponent_player
        self.include_opponent_turn = config.include_opponent_turn

        BaseDiscreteActionEnv.__init__(self)

        # Load Contract Bridge game
        self.game_parameters = {
            "dealer": config.dealer,
            "dealer_vul": config.dealer_vul,
            "non_dealer_vul": config.non_dealer_vul,
            "use_double_dummy_result": config.use_double_dummy_result,
        }
        self._env = pyspiel.load_game("bridge", self.game_parameters)
        self.num_players = 4

        self.state = None
        self.num_steps = 0
        self.history = []
        self.player_cards = {}  # Store dealt cards for each player

        # [Correction 2] Team Logic Adjustment
        # North(0) and South(2) are partners. East(1) and West(3) are opponents.
        # If we control Player 0 (North), Player 2 (South) is our partner.
        self.controlled_player = 0
        # By default, treat East(1) and West(3) as opponents to be handled by built-in logic.
        # South(2) is excluded from 'opponents' to allow for potential multi-agent control or specific partner modeling.
        # (If South is not controlled by an external agent, it will fall through to random/rule-based in _opponent_step if added to this list)
        self.opponent_players = [1, 3] 

    @property
    def current_player(self):
        if self.state is None:
            return 0
        if self.state.is_terminal():
            return 0
        return self.state.current_player()

    def _is_opponent(self, player_id: int) -> bool:
        """Check if the given player is an opponent."""
        if self.built_in_opponent == "none":
            return False
        return player_id in self.opponent_players

    def _detect_phase(self) -> str:
        """Detects if the game is in Bidding or Play phase based on history."""
        # In OpenSpiel Bridge, Bidding ends after 3 consecutive passes (except initially).
        # A simple heuristic: check if the state is terminal or count passes.
        # A more robust way in OpenSpiel is checking the move history length or specific state flags if exposed.
        # Here we rely on the Action ID. Bidding actions are < 52 (usually). Play actions are cards (52+).
        # However, OpenSpiel Bridge IDs: Pass=0, Dbl=1, RDbl=2, Bid=3-37. Play=52+.
        # If any action >= 52 has occurred, we are definitely in Play phase.
        
        # Check history for any card play action
        for action_str in self.history:
            # This is a string check, slightly fragile but works if format is consistent
            if "Pass" in action_str or "Double" in action_str or "NT" in action_str or any(s in action_str for s in ['♣', '♦', '♥', '♠']):
                # These are bidding strings mostly, but cards also have suits.
                # Let's check the underlying state history if possible, or search for Play cues.
                pass
                
        # Better approach: Check if the last sequence of actions was 3 Passes (and history len > 3 or 4)
        if len(self.history) > 3:
            last_three = [h.split(":")[-1].strip() for h in self.history[-3:]]
            if all(a == "<Pass>" for a in last_three):
                # Check if it wasn't the initial Pass-out
                if len(self.history) == 4 and self.history[0].split(":")[-1].strip() == "<Pass>":
                     return "Bidding" # Initial 4 passes = pass out (terminal)
                return "Play"
        
        # Fallback: check if we have played any cards yet
        # (This logic can be refined by tracking the 'contract' state)
        return "Bidding"

    def reset(self, seed: Optional[int] = 0):
        try:
            with all_seed(seed):
                self.state = self._env.new_initial_state()
                self.num_steps = 0
                self.history = []

                # Handle chance nodes (dealing cards)
                while self.state.is_chance_node():
                    outcomes_with_probs = self.state.chance_outcomes()
                    actions, probs = zip(*outcomes_with_probs)
                    action = random.choices(actions, weights=probs)[0]
                    self.state.apply_action(action)

                # Store player observations
                for p in range(self.num_players):
                    # [Correction 1] Always use information_state_string for richer info (history + private cards)
                    self.player_cards[p] = self.state.information_state_string(p)

                initial_observation = {
                    'observation': self.render(),
                    'legal_actions': self.get_all_actions(),
                }
                execute_results = []
                
                # Let opponents take actions if needed
                if self.built_in_opponent != "none":
                    done = self.state.is_terminal()
                    while self._is_opponent(self.current_player) and not done:
                        current_player = self.current_player
                        opponent_action = self._opponent_step()
                        observation, rewards, done, info = self._step(opponent_action)
                        execute_results.append({
                            'current_player': current_player,
                            'action': self._action_to_string(current_player, opponent_action),
                            'rewards': rewards,
                            'done': done,
                            'info': info,
                            'next_player': self.current_player,
                            'observation': observation,
                            'legal_actions': self.get_all_actions(),
                        })
                return initial_observation, execute_results
        except (RuntimeError, RuntimeWarning) as e:
            next_seed = abs(hash(str(seed))) % (2**32) if seed is not None else 0
            return self.reset(next_seed)

    def step(self, action):
        execute_results = []
        current_player = self.current_player
        observation, rewards, done, info = self._step(action)

        execute_results.append({
            'current_player': current_player,
            'action': self._action_to_string(current_player, action),
            'rewards': rewards,
            'done': done,
            'info': info,
            'next_player': self.current_player,
            'observation': observation,
            'legal_actions': self.get_all_actions(),
        })
        
        # If chose to play with built-in opponent, let opponents take actions
        if self.built_in_opponent != "none":
            # Note: If self.opponent_players only includes [1, 3], 
            # Player 2 (Partner) actions must be handled externally or added to opponents list.
            while self._is_opponent(self.current_player) and not done:
                current_player = self.current_player
                opponent_action = self._opponent_step()
                observation, rewards, done, info = self._step(opponent_action)
                execute_results.append({
                    'current_player': current_player,
                    'action': self._action_to_string(current_player, opponent_action),
                    'rewards': rewards,
                    'done': done,
                    'info': info,
                    'next_player': self.current_player,
                    'observation': observation,
                    'legal_actions': self.get_all_actions(),
                })
        return execute_results

    def _step(self, action):
        if isinstance(action, str):
            action = self._string_to_action(action)
        if self.state is None or self.state.is_terminal():
            raise RuntimeError("Cannot apply action on a terminal state.")

        current_player = self.current_player
        action_str = self._action_to_string(current_player, action)
        player_name = self.PLAYER_NAMES[current_player]
        
        self.history.append(f"{player_name}: {action_str}")
        self.state.apply_action(action)
        self.num_steps += 1

        observation = self.render()
        rewards = list(self.state.rewards())
        done = self.state.is_terminal()
        info = self._get_info()

        return observation, rewards, done, info

    def _opponent_step(self):
        # [Improvement] Better random logic or heuristic could go here
        if self.built_in_opponent == "random":
            legal_actions = list(self.get_all_actions().keys())
            action = random.choice(legal_actions)
        else:
            raise ValueError(f"Invalid built-in opponent: {self.built_in_opponent}")
        return action

    def get_prompt(self, mode="prefix", think=True, player_id=0):
        if mode == "prefix":
            prefix_prompt = self._get_prefix_prompt(think, player_id)
            return prefix_prompt
        else:
            raise ValueError(f"Invalid prompt mode: {mode}")

    def _get_prefix_prompt(self, think=True, player_id=0):
        system_prompt = "You are an AI agent that makes optimal decisions in Contract Bridge to maximize your team's score."

        player_name = self.PLAYER_NAMES[player_id]
        partner_id = (player_id + 2) % 4
        partner_name = self.PLAYER_NAMES[partner_id]
        
        if player_id in [0, 2]:
            team = "North-South"
            opponents = "East-West"
        else:
            team = "East-West"
            opponents = "North-South"

        rules = (
            "1. Contract Bridge is a 4-player card game with two teams: North-South vs East-West.\n"
            "2. A standard 52-card deck is used. Each player receives 13 cards.\n"
            "3. The game has two phases: Bidding and Play.\n"
            "\n"
            "BIDDING PHASE:\n"
            "4. Players bid in clockwise order (North, East, South, West).\n"
            "5. A bid consists of a level (1-7) and a denomination (♣, ♦, ♥, ♠, or NT).\n"
            "   - Suits rank: ♣ < ♦ < ♥ < ♠ < NT\n"
            "   - Higher level always beats lower level\n"
            "6. Special bids: Pass, Double (X), Redouble (XX)\n"
            "7. Bidding ends after three consecutive passes following a bid.\n"
            "8. The final bid becomes the 'contract' that the declaring team must fulfill.\n"
            "\n"
            "PLAY PHASE:\n"
            "9. The declarer's partner (dummy) lays cards face up.\n"
            "10. Players play one card each in clockwise order (a 'trick').\n"
            "11. **IMPORTANT**: If you are the Declarer, you also control the Dummy's hand. When it is Dummy's turn, you must select the card for them.\n"
            "12. The team that wins enough tricks based on their contract scores points."
        )

        information = (
            f"1. You are {player_name}. Your partner is {partner_name}.\n"
            f"2. Your team is {team}. Your opponents are {opponents}.\n"
            "3. In each turn, you will see:\n"
            "   a. Your 13 cards\n"
            "   b. The bidding/play history\n"
            "   c. Available legal actions\n"
            "   d. **The Dummy's exposed cards** (Only during Play Phase)\n"
            "4. During bidding, communicate your hand strength through bids.\n"
            "5. During play, coordinate with your partner to win tricks."
        )

        FORMAT_PROMPT = "<answer>{your chosen action}</answer>"
        FORMAT_PROMPT_EXAMPLE = "<answer><1NT></answer>"

        instructions = (
            f"Always choose only one action from the legal actions and output `{FORMAT_PROMPT}` with no extra text after you finish the thinking process. "
            f"For example, `{FORMAT_PROMPT_EXAMPLE}`. "
            "Strictly follow the above format and keep your thinking process concise. Responses that do not follow the format will result in a penalty."
        )

        user_prompt = (
            f"GAME RULES:\n{rules}\n\n"
            f"PLAYER INFORMATION:\n{information}\n\n"
            f"RESPONSE INSTRUCTIONS:\n{instructions}\n\n"
        )
        prefix_prompt = {"system": system_prompt, "user": user_prompt}
        return prefix_prompt

    def get_all_actions(self):
        return self._get_legal_actions(self.current_player)

    def _get_legal_actions(self, player_id):
        legal_actions = dict()
        if self.state is None:
            return legal_actions
        actions = self.state.legal_actions(player_id)
        for a in actions:
            legal_actions[a] = self._action_to_string(player_id, a)
        return legal_actions

    def _action_to_string(self, player_id, action):
        if isinstance(action, str):
            return action
        action_str = self.state.action_to_string(player_id, action)
        return f"<{action_str}>"

    def _string_to_action(self, action_str):
        action_str = action_str.strip()
        action_str = action_str.replace("<", "").replace(">", "")
        return self.state.string_to_action(action_str)

    def _get_info(self):
        info = {}
        if self.state.is_terminal():
            returns = self.state.returns()
            info.update({
                "player_0_return": returns[0],
                "player_1_return": returns[1],
                "player_2_return": returns[2],
                "player_3_return": returns[3],
                # Add default zero values for error metrics
                **{f"player_{i}_lose_for_wrong_format": 0 for i in range(4)},
                **{f"player_{i}_lose_for_overlong_response": 0 for i in range(4)},
                **{f"player_{i}_lose_for_overlong_sequence": 0 for i in range(4)},
            })
        return info

    def get_losing_state(self, player_id: int = 0, overlong_response: bool = False, overlong_sequence: bool = False):
        observation = self.render()
        done = True
        
        returns = self.state.returns() if self.state.is_terminal() else [0, 0, 0, 0]
        reward = [0, 0, 0, 0]
        reward[player_id] = -abs(returns[player_id]) - 10
        
        info = {
            "player_0_return": returns[0],
            "player_1_return": returns[1],
            "player_2_return": returns[2],
            "player_3_return": returns[3],
        }
        for i in range(4):
            info[f"player_{i}_lose_for_wrong_format"] = 1 if i == player_id else 0
            info[f"player_{i}_lose_for_overlong_response"] = (1 if overlong_response else 0) if i == player_id else 0
            info[f"player_{i}_lose_for_overlong_sequence"] = (1 if overlong_sequence else 0) if i == player_id else 0

        execute_results = [{
            'current_player': player_id,
            'action': '',
            'rewards': reward,
            'done': done,
            'info': info,
            'next_player': None,
            'observation': None,
            'legal_actions': None,
        }]
        return execute_results

    def render(self, mode: str = "text"):
        if mode == "text":
            return self._render_text()
        elif mode == "rgb_array":
            return self._render_rgb_array()
        else:
            raise ValueError(f"Invalid mode: {mode}")

    def _render_text(self):
        if self.state.is_terminal():
            return "The game is complete."

        current_player = self.current_player
        
        # [Correction 1 & 3] Use information_state_string and detect phase
        try:
            # information_state_string typically contains private cards + public history
            obs = self.state.information_state_string(current_player)
        except:
            obs = self.state.observation_string(current_player)
        
        phase = self._detect_phase()
        
        obs_str = []
        obs_str.append(f"Current Phase: {phase}")
        obs_str.append(f"Current Player (Turn): {self.PLAYER_NAMES[current_player]}")
        
        if phase == "Play":
             obs_str.append("[Note: If you are the Declarer and this is Dummy's turn, you must play for them.]")
             # In a real OpenSpiel observation, the Dummy's cards are usually appended to the string 
             # if the observer is the declarer. 
        
        obs_str.append("-" * 20)
        obs_str.append("Observation State:")
        obs_str.append(obs)
        
        if self.history:
            obs_str.append("-" * 20)
            obs_str.append("Recent Actions History:")
            for action in self.history[-10:]:
                obs_str.append(f"  {action}")
        
        return "\n".join(obs_str)

    def _render_rgb_array(self):
        warnings.warn("Bridge does not support image rendering yet.")
        return None

    def close(self):
        """Close the environment."""
        if hasattr(self, "_env") and self._env is not None:
            pass


if __name__ == "__main__":
    import sys
    
    print("=" * 100)
    print("Testing Contract Bridge (Modified)")
    print("=" * 100)
    
    config = BridgeConfig(built_in_opponent="random")
    env = Bridge(config)
    
    prefix_prompt = env.get_prompt(mode="prefix", player_id=0)
    print(f"System prompt sample:\n{prefix_prompt['system'][:200]}...\n")
    
    player_returns = {i: [] for i in range(4)}
    
    for i in range(1): # Reduced to 1 for brevity in test
        print('-' * 100)
        print(f'Episode {i}')
        print('-' * 100)
        
        initial_observation, execute_results = env.reset(seed=i)
        
        done = False
        steps = 0
        while not done:
            # Simple random step for testing
            legal_actions = env.get_all_actions()
            action = random.choice(list(legal_actions.values()))
            
            # Print current state info to verify rendering
            if steps % 10 == 0: 
                print(f"\n[Step {steps}] {env._render_text().split('Observation State:')[0]}...")
            
            execute_result = env.step(action)
            done = execute_result[-1]['done']
            steps += 1
            
        info = execute_result[-1]['info']
        print(f"\nGame finished in {steps} steps.")
        print(f"Returns: {[info[f'player_{i}_return'] for i in range(4)]}")

    print("\n" + "=" * 100)
    print("Test completed.")
    print("=" * 100)