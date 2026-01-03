import matplotlib.pyplot as plt
import numpy as np
import os
import random
from roll.agentic.env.tiny_bridge.config import TinyBridgeConfig
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

class TinyBridge(BaseDiscreteActionEnv):
    """
    Tiny Bridge (4-Player) Full Game Environment
    
    Structure:
    - Phase 1: Auction (handled by OpenSpiel tiny_bridge_4p logic)
    - Phase 2: Play (handled by Python logic)
    
    Players:
    - 0 (North, Agent 1) & 2 (South, Agent 2): Controlled by User/LLM (Team Us)
    - 1 (East, Bot) & 3 (West, Bot): Controlled by Environment (Team Opponent)
    """

    # Tiny Bridge Constants
    SUITS = ['H', 'S'] # Hearts, Spades
    RANKS = ['J', 'Q', 'K', 'A']
    TRUMP_ORDER = ['H', 'S', 'N'] # Hearts, Spades, No-Trump
    
    # Action Mapping
    # OpenSpiel Actions: 0(Pass), 1(1H)..6(2NT), 7(Dbl), 8(RDbl)
    BID_ACTIONS = {
        0: "Pass", 1: "1H", 2: "1S", 3: "1NT", 
        4: "2H", 5: "2S", 6: "2NT", 7: "Dbl", 8: "RDbl"
    }
    # Reverse Mapping for execution
    ACTION_TO_ID = {v: k for k, v in BID_ACTIONS.items()}
    
    PLAYER_NAMES = {0: "North (Agent)", 1: "East (Bot)", 2: "South (Agent)", 3: "West (Bot)"}

    def __init__(self, config: TinyBridgeConfig = TinyBridgeConfig()):
        self.config = config
        self.built_in_opponent = getattr(config, "built_in_opponent", "none") 
        self.opponent_player = getattr(config, "opponent_player", 1) # 기본값 설정
        self.include_opponent_turn = getattr(config, "include_opponent_turn", "action")
        self.render_mode = config.render_mode
        BaseDiscreteActionEnv.__init__(self)

        # Load 4-player game for Auction Phase
        self._auction_env = pyspiel.load_game("tiny_bridge_4p")
        self.num_players = 4
        
        # Team Configuration
        self.agents = [0, 2]  # North, South
        self.bots = [1, 3]    # East, West
        
        # Internal State
        self.phase = "Start" # Start, Auction, Play, Terminal
        self.hands = {i: [] for i in range(4)} # Player hands
        self.auction_state = None
        
        # Play Phase State
        self.contract = None 
        self.tricks_won = {0: 0, 1: 0} # 0 for Team NS, 1 for Team EW
        self.current_trick = [] # List of (player, card_str)
        self.leader = 0 # Who leads the current trick
        self.trump_suit = None
        self.history = deque(maxlen=20)
        self.num_steps = 0

    @property
    def current_player(self):
        """
        Expose a 2-player view (agent vs opponent) to match EnvManager expectations.
        Agent seats: 0 (North) and 2 (South) -> agent_id 0 and 1 respectively.
        Opponent seats: 1 (East) and 3 (West) -> opponent_id 1 (shared).
        """
        seat = self._current_seat()
        return self._seat_to_agent_id(seat)

    def _current_seat(self):
        if self.phase == "Auction":
            if self.auction_state:
                return self.auction_state.current_player()
        elif self.phase == "Play":
            if not self.current_trick:
                return self.leader
            else:
                last_player = self.current_trick[-1][0]
                return (last_player + 1) % 4
        return 0 

    def _seat_to_agent_id(self, seat: int) -> int:
        # Map seats to 2-player view: agents (0,2) -> 0/1; opponents (1,3) -> 1
        if seat in self.agents:
            return 0 if seat == self.agents[0] else 1
        return 1  # any opponent seat collapses to opponent id

    def _agent_id_to_seat(self, agent_id: int) -> int:
        # Map agent view back to seat: 0 -> North (0), 1 -> South (2)
        return self.agents[agent_id % 2]

    def _agent_id_matches_seat(self, agent_id: int, seat: int) -> bool:
        return (agent_id == 0 and seat in self.agents[:1] + self.agents[1:2]) or (
            agent_id == 1 and (seat in self.agents[1:2] or seat in self.bots)
        )

    def reset(self, seed: Optional[int] = 0):
        try:
            with all_seed(seed):
                # 1. Start Auction Phase
                self.phase = "Auction"
                self.auction_state = self._auction_env.new_initial_state()
                self.history.clear()
                self.tricks_won = {0: 0, 1: 0}
                self.current_trick = []
                self.num_steps = 0
                self.hands = {i: [] for i in range(4)}
                
                # 2. Deal Cards (Handle Chance Nodes)
                while self.auction_state.is_chance_node():
                    outcomes = self.auction_state.chance_outcomes()
                    actions, probs = zip(*outcomes)
                    action = random.choices(actions, weights=probs)[0]
                    self.auction_state.apply_action(action)
                
                # 3. Parse Hands
                self._parse_hands(self.auction_state.to_string())
                
                # 4. Progress game until an Agent's turn
                observation, rewards, done, info = self._progress_game()
                
                initial_observation = {
                    'observation': observation,
                    'legal_actions': self.get_all_actions(),
                }
                
                # Execute results logic (similar to Hanabi, capturing the initial transition)
                execute_results = [{
                    'current_player': self.current_player,
                    'action': None,
                    'rewards': rewards,
                    'done': done,
                    'info': info,
                    'next_player': self.current_player if not done else None,
                    'observation': observation,
                    'legal_actions': self.get_all_actions() if not done else {},
                }]
                
                return initial_observation, execute_results

        except (RuntimeError, RuntimeWarning) as e:
            # Fallback for seed issues
            next_seed = abs(hash(str(seed))) % (2**32) if seed is not None else 0
            return self.reset(next_seed)

    def step(self, action):
        execute_results = []
        current_player = self.current_player  # agent-view id (0/1)
        seat = self._current_seat()           # actual seat index 0-3
        
        # 1. Agent Step
        observation, rewards, done, info = self._step(action, seat)

        execute_results.append({
            'current_player': current_player,
            'action': self._action_to_string(current_player, action),
            'rewards': rewards,
            'done': done,
            'info': info,
            'next_player': self.current_player if not done else None,
            'observation': observation,
            'legal_actions': self.get_all_actions() if not done else {},
        })
        
        # 2. Progress through Bot turns if any
        # bots are collapsed into opponent id (1) in the current_player view
        if not done and self._current_seat() in self.bots:
             # _progress_game handles the loop of bot actions
             observation, rewards, done, info = self._progress_game()
             
             # Append the state AFTER bots have moved (when control returns to agent or game ends)
             execute_results.append({
                'current_player': self.current_player if not done else None,
                'action': "Bots Played", # Placeholder
                'rewards': rewards,
                'done': done,
                'info': info,
                'next_player': self.current_player if not done else None,
                'observation': observation,
                'legal_actions': self.get_all_actions() if not done else {},
            })

        return execute_results

    def _step(self, action, seat: int = None):
        if self.phase == "Terminal":
             raise RuntimeError("Cannot apply action on a terminal state.")

        if seat is None:
            seat = self._current_seat()

        # Convert string to ID/format if needed
        if isinstance(action, str):
            clean_action = action.replace("<", "").replace(">", "").strip()
            if self.phase == "Auction":
                if clean_action in self.ACTION_TO_ID:
                    action = self.ACTION_TO_ID[clean_action]
                else:
                    # Fallback or error
                    try:
                        action = self._auction_env.string_to_action(clean_action)
                    except:
                        pass # Will likely fail in apply_action
            else:
                action = clean_action

        # Log action
        player_name = self.PLAYER_NAMES[seat]
        action_str = self._action_to_string(seat, action)
        self.history.append(f"{player_name}: {action_str}")

        # Apply Action
        self._apply_action_logic(seat, action)
        self.num_steps += 1

        # Check Phase Transitions (Auction -> Play) happens inside _apply_action_logic
        
        observation = self.render()
        rewards = self._get_current_rewards()
        done = (self.phase == "Terminal")
        info = self._get_info()

        return observation, rewards, done, info

    def _progress_game(self):
        """
        Advances the game through Bot turns until it is an Agent's turn or the game ends.
        """
        while self._current_seat() in self.bots and self.phase != "Terminal":
            seat = self._current_seat()
            bot_action = self._get_bot_action(seat)
            
            player_name = self.PLAYER_NAMES[seat]
            self.history.append(f"{player_name}: {bot_action}")
            
            clean_action = self._string_to_action(bot_action)

            if self.phase == "Auction":
                # Auction: ID로 변환 필요
                if clean_action in self.ACTION_TO_ID:
                    action_id = self.ACTION_TO_ID[clean_action]
                    self._apply_action_logic(seat, action_id)
                else:
                    try:
                        action_id = self._auction_env.string_to_action(clean_action)
                        self._apply_action_logic(seat, action_id)
                    except:
                        # Fallback: Pass (0)
                        self._apply_action_logic(seat, 0)
            else:
                # Play: 문자열 그대로 사용 (예: "HK")
                # hands 리스트에 있는 값과 일치해야 하므로 clean_action 사용
                self._apply_action_logic(seat, clean_action)
                
            self.num_steps += 1

        observation = self.render()
        rewards = self._get_current_rewards()
        done = (self.phase == "Terminal")
        info = self._get_info()
        
        return observation, rewards, done, info

    def _apply_action_logic(self, player, action):
        if self.phase == "Auction":
            # action should be int ID here
            self.auction_state.apply_action(action)
            if self.auction_state.is_terminal():
                self._transition_to_play()
                
        elif self.phase == "Play":
            # action should be string here (e.g. "HA")
            self.current_trick.append((player, action))
            if action in self.hands[player]:
                self.hands[player].remove(action)
            
            if len(self.current_trick) == 4:
                self._resolve_trick()

    def _transition_to_play(self):
        self.phase = "Play"
        history = self.auction_state.history()
        bids = [a for a in history if a < 100] 
        
        last_bid = 0 
        declarer = -1
        doubled = False
        redoubled = False
        pass_count = 0
        bid_history = [] 
        current_bidder = 0 
        
        for action in bids:
            if action == 0: pass_count += 1
            elif action == 7: 
                doubled = True
                pass_count = 0
            elif action == 8:
                doubled = False
                redoubled = True
                pass_count = 0
            else:
                last_bid = action
                doubled = False
                redoubled = False
                pass_count = 0
                bid_history.append((current_bidder, action))
            current_bidder = (current_bidder + 1) % 4
            
        if last_bid == 0:
            self.phase = "Terminal"
            self.contract = "Passed Out"
            return

        suit_idx = (last_bid - 1) % 3 
        trump_char = self.TRUMP_ORDER[suit_idx]
        self.trump_suit = trump_char
        
        winning_team_mod = bid_history[-1][0] % 2
        for p, bid in bid_history:
            if p % 2 == winning_team_mod and (bid - 1) % 3 == suit_idx:
                declarer = p
                break
        
        self.contract = {
            'bid_val': last_bid, 'trump': trump_char, 'declarer': declarer,
            'doubled': doubled, 'redoubled': redoubled
        }
        self.leader = (declarer + 1) % 4
        contract_str = f"{self.BID_ACTIONS[last_bid]} by {self.PLAYER_NAMES[declarer]}"
        self.history.append(f"--- Auction End. Contract: {contract_str} ---")

    def _resolve_trick(self):
        lead_suit = self.current_trick[0][1][1] 
        winner_idx = 0
        best_card = self.current_trick[0][1]
        
        for i in range(1, 4):
            curr_player, curr_card = self.current_trick[i]
            curr_suit = curr_card[1]
            best_suit = best_card[1]
            
            is_curr_trump = (curr_suit == self.trump_suit)
            is_best_trump = (best_suit == self.trump_suit)
            
            if is_curr_trump and not is_best_trump:
                winner_idx = i
                best_card = curr_card
            elif is_curr_trump and is_best_trump:
                if self._rank_val(curr_card) > self._rank_val(best_card):
                    winner_idx = i
                    best_card = curr_card
            elif not is_curr_trump and not is_best_trump:
                if curr_suit == lead_suit:
                     if self._rank_val(curr_card) > self._rank_val(best_card):
                        winner_idx = i
                        best_card = curr_card
                        
        winner_player = self.current_trick[winner_idx][0]
        winning_team = winner_player % 2 
        
        self.tricks_won[winning_team] += 1
        self.leader = winner_player
        
        trick_str = ", ".join([f"{self.PLAYER_NAMES[p].split()[0]}:{c}" for p, c in self.current_trick])
        self.history.append(f"Trick Result: [{trick_str}] -> Winner: {self.PLAYER_NAMES[winner_player]}")
        
        self.current_trick = []
        if not any(self.hands.values()):
            self.phase = "Terminal"

    def _get_bot_action(self, player):
        legal = self._get_legal_actions(player)
        if not legal: return "Pass"
        actions = list(legal.values())
        return random.choice(actions)

    def get_prompt(self, mode="prefix", think=True, player_id=0):
        if mode == "prefix":
            return self._get_prefix_prompt(think, player_id)
        else:
            raise ValueError(f"Invalid prompt mode: {mode}")

    def _get_prefix_prompt(self, think=True, player_id=0):
        # player_id is the 2-player view (0->seat 0, 1->seat 2)
        seat = self._agent_id_to_seat(player_id)
        partner_seat = 2 if seat == 0 else 0
        partner_id = 0 if partner_seat == 0 else 1
        
        system_prompt = "You are an AI agent playing Tiny Bridge. Your goal is to cooperate with your partner to maximize your team's score."
        
        rules = (
            "1. **Tiny Bridge** is a simplified 4-player version of Contract Bridge.\n"
            f"2. **Players**: You are {self.PLAYER_NAMES[seat]}. Your partner is {self.PLAYER_NAMES[partner_seat]}. "
            "The other two are opponents.\n"
            "3. **Deck**: 8 cards total (Hearts H, Spades S / Ranks J, Q, K, A).\n"
            "4. **Phase 1: Auction**: Players bid to set the contract (Trumps & Level).\n"
            "   - Bids: 1H, 1S, 1NT (Need 1 trick), 2H, 2S, 2NT (Need 2 tricks).\n"
            "   - Special: Double, Redouble, Pass.\n"
            "5. **Phase 2: Play**: 2 Tricks are played.\n"
            "   - Must follow the suit of the led card if possible.\n"
            "   - High card wins unless Trumped.\n"
            "6. **Scoring**: +10/+30/+35 based on contract. Penalties for failing.\n"
        )
        
        information = (
            f"1. You are playing as {self.PLAYER_NAMES[player_id]}.\n"
            "2. In each turn, you will be provided with your hand, the auction history, and the play history.\n"
            "3. During play, observe the current trick and legal moves (you must follow suit!).\n"
        )
        
        FORMAT_PROMPT = "<answer>{your chosen action}</answer>"
        FORMAT_PROMPT_EXAMPLE = "<answer><1H></answer>"
        
        instructions = (
            f"Always choose only one action from the legal actions and output `{FORMAT_PROMPT}`. "
            f"Example: `{FORMAT_PROMPT_EXAMPLE}`. "
            "Strictly follow the format. Responses that do not follow the format will result in penalties."
        )
        
        user_prompt = (
            f"GAME RULES:\n{rules}\n\n"
            f"PLAYER INFORMATION:\n{information}\n\n"
            f"RESPONSE INSTRUCTIONS:\n{instructions}\n\n"
        )
        prefix_prompt = {"system": system_prompt, "user": user_prompt}
        return prefix_prompt

    def get_all_actions(self):
        return self._get_legal_actions()

    def _get_legal_actions(self, player=None):
        """
        Return legal actions for the given player.
        player can be:
        - None: use current seat
        - seat index (0-3)
        - agent-view id (0/1), which will be mapped to seat
        """
        if player is None:
            seat = self._current_seat()
        elif player in [0, 1, 2, 3]:
            seat = player
        else:
            seat = self._agent_id_to_seat(player)
        actions = {}
        if self.phase == "Auction":
            legal_ids = self.auction_state.legal_actions(seat)
            for aid in legal_ids:
                actions[aid] = self._action_to_string(seat, aid)
        elif self.phase == "Play":
            hand = self.hands[seat]
            if not self.current_trick:
                for idx, card in enumerate(hand):
                    actions[idx] = self._action_to_string(seat, card)
            else:
                lead_suit = self.current_trick[0][1][1]
                follow_cards = [c for c in hand if c[1] == lead_suit]
                candidates = follow_cards if follow_cards else hand
                for idx, card in enumerate(candidates):
                    actions[idx] = self._action_to_string(seat, card)
        return actions

    def _action_to_string(self, player_id, action):
        # Format action for Agent consumption (adding brackets)
        if isinstance(action, str):
            # Already a string (Play phase card), just wrap it
            return f"<{action}>" if not action.startswith("<") else action
        
        # ID (Auction phase) -> String
        if action in self.BID_ACTIONS:
            return f"<{self.BID_ACTIONS[action]}>"
        
        # Fallback for OpenSpiel internal actions
        return f"<{action}>"

    def _string_to_action(self, action_str):
        # Remove brackets for internal processing
        return action_str.replace("<", "").replace(">", "").strip()

    def _get_info(self):
        info = {}
        if self.phase == "Terminal":
            rewards = self._get_current_rewards()
            info = {
                "player_0_return": rewards[0],
                "player_1_return": rewards[1], # Bot
                "player_2_return": rewards[2],
                "player_3_return": rewards[3], # Bot
                # Error flags placeholders
                "player_0_lose_for_wrong_format": 0,
                "player_2_lose_for_wrong_format": 0,
            }
        return info

    def _get_current_rewards(self):
        if self.phase != "Terminal":
            return [0.0] * 4
        return self._calculate_score()

    def get_losing_state(self, player_id: int=0, overlong_response: bool=False, overlong_sequence: bool=False):
        # Handle format errors/timeouts
        observation = self.render()
        done = True
        
        # Penalty score
        reward = [0.0] * 4
        reward[player_id] = -100.0 # Heavy penalty
        
        info = {
            "player_0_return": reward[0],
            "player_2_return": reward[2],
            "player_0_lose_for_wrong_format": 1 if player_id == 0 else 0,
            "player_2_lose_for_wrong_format": 1 if player_id == 2 else 0,
            "player_0_lose_for_overlong_response": 1 if overlong_response and player_id == 0 else 0,
            "player_2_lose_for_overlong_response": 1 if overlong_response and player_id == 2 else 0,
        }
        
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

    def _calculate_score(self):
        if self.contract == "Passed Out": return [0.0] * 4
        
        declarer_team = self.contract['declarer'] % 2 
        tricks_needed = 1 + (self.contract['bid_val'] - 1) // 3
        if self.contract['bid_val'] > 3: tricks_needed = 2
        
        tricks_got = self.tricks_won[declarer_team]
        score = 0
        if tricks_got >= tricks_needed:
            bid_level = (self.contract['bid_val'] - 1) // 3 + 1
            suit = self.contract['trump']
            if bid_level == 1: score = 10
            elif bid_level == 2: score = 35 if suit == 'N' else 30
            if tricks_got > tricks_needed: score += 10
        else:
            score = -20 * (tricks_needed - tricks_got)
            
        if self.contract['doubled']: score *= 2
        if self.contract['redoubled']: score *= 4
        
        rewards = [0.0] * 4
        if declarer_team == 0: # NS
            rewards[0] = float(score); rewards[2] = float(score)
            rewards[1] = float(-score); rewards[3] = float(-score)
        else: # EW
            rewards[0] = float(-score); rewards[2] = float(-score)
            rewards[1] = float(score); rewards[3] = float(score)
        return rewards

    def render(self, mode: str = "text"):
        if mode == "text":
            return self._render_text()
        return None

    def _render_text(self):
        if self.phase == "Terminal":
            return "The game is over."
            
        curr_seat = self._current_seat()
        player_name = self.PLAYER_NAMES[curr_seat]
        
        obs_str = [
            f"1. Current Phase: {self.phase}",
            f"2. You are: {player_name}",
            f"3. Contract: {self.contract if self.contract else 'Bidding in progress...'}",
            f"4. Tricks Won: NS (Us) {self.tricks_won[0]} - EW (Opp) {self.tricks_won[1]}",
        ]
        
        # Hands
        if curr_seat in self.hands:
             obs_str.append(f"5. Your Hand: {self.hands[curr_seat]}")
        
        # Play info
        if self.phase == "Play":
            obs_str.append(f"6. Trump Suit: {self.trump_suit}")
            trick_str = ", ".join([f"{self.PLAYER_NAMES[p].split()[0]}:{c}" for p, c in self.current_trick])
            obs_str.append(f"7. Current Trick: [{trick_str}]")
            
        # History
        obs_str.append(f"8. Recent History:")
        for h in list(self.history)[-5:]:
             obs_str.append(f"    - {h}")
             
        return "\n".join(obs_str)

    def _rank_val(self, card_str):
        # 'HA' -> 'A' -> 3
        rank_char = card_str[1]
        return {'J':0, 'Q':1, 'K':2, 'A':3}[rank_char]

    def _parse_hands(self, state_str):
        parts = state_str.split(" ")
        for part in parts:
            if ":" in part:
                seat_char, cards = part.split(":")
                if seat_char in ['N', 'E', 'S', 'W']:
                    seat_idx = {'N':0, 'E':1, 'S':2, 'W':3}[seat_char]
                    self.hands[seat_idx] = [cards[i:i+2] for i in range(0, len(cards), 2)]
                    
    def _get_player_name(self, player_id):
        return self.PLAYER_NAMES[player_id]

    def close(self):
        if hasattr(self, "_auction_env") and self._auction_env is not None:
            pass


if __name__ == "__main__":
    print("-" * 100)
    print("TinyBridge Unit Test")
    print("-" * 100)
    env = TinyBridge()
    
    for i in range(3):
        print(f"\n[Episode {i}]")
        prefix_prompt = env.get_prompt(mode="prefix")
        # print(f"System Prompt Snippet: {prefix_prompt['system'][:100]}...")
        
        initial_observation, execute_results = env.reset(seed=i)
        
        # Determine starting point from results or initial obs
        if execute_results:
             current_obs = execute_results[-1]['observation']
             current_legal = execute_results[-1]['legal_actions']
        else:
             current_obs = initial_observation['observation']
             current_legal = initial_observation['legal_actions']
             
        done = False
        step_count = 0
        
        while not done:
            print(f"\n--- Step {step_count} (Player {env.current_player}) ---")
            print(f"Observation:\n{current_obs}")
            print(f"Legal Actions: {current_legal}")
            
            if not current_legal:
                break
                
            action_str = random.choice(list(current_legal.values()))
            print(f">> Selected Action: {action_str}")
            
            results = env.step(action_str)
            
            # Analyze results (last item represents final state after bots moved)
            last_result = results[-1]
            current_obs = last_result['observation']
            current_legal = last_result['legal_actions']
            done = last_result['done']
            info = last_result['info']
            rewards = last_result['rewards']
            
            step_count += 1
            
        print(f"Game Over. Rewards: {rewards}")
        print("-" * 50)