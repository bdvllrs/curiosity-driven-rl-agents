import os
import random
import numpy as np
import imageio
import cv2
import matplotlib.pyplot as plt
from utils import maze, config

colors = {
    "coin": (0.9, 0.8, 0.25),
    "agent": (0.25, 0.9, 0.9)
}


class Env:
    def __init__(self):
        self.board = None
        self.scale = 16
        self.episode = 0
        self.iter = 0
        self.size = config().sim.env.size
        self.agent_position = None
        self.coin_positions = None
        self.board_memory = []
        self.state_memory = []
        self.mask_board = None
        self.seen_positions = []
        self.max_length = config().sim.env.max_length

    def _get_board(self):
        board = self.board.copy()
        x, y = self.agent_position[0], self.agent_position[1]
        board[x, y] = 0.6
        for coord in self.coin_positions:
            x, y = coord
            board[x, y] = 0.3
        return board

    def _add_board(self, board):
        self.board_memory.append(self._board_to_image(board))

    def _board_to_image(self, board):
        if len(board.shape) == 2:
            state = np.zeros((board.shape[0], board.shape[1], 3))
            for i in range(board.shape[0]):
                for j in range(board.shape[1]):
                    if board[i, j] == 0:  # wall
                        state[i, j, :] = [0, 0, 0]
                    elif board[i, j] == 1:  # free position
                        state[i, j, :] = [1, 1, 1]
                    elif board[i, j] == 0.3:  # coin
                        state[i, j, :] = colors["coin"]
                    else:  # agent
                        state[i, j, :] = colors["agent"]
            return state
        return board

    def _add_state(self, board):
        self.state_memory.append(self._board_to_image(board))

    def _get_state_hard(self, board, blank):
        x, y = self.agent_position
        stop_right = False
        stop_left = False
        stop_top = False
        stop_bottom = False
        state = np.zeros_like(board)
        if blank:
            return state
        state[x, y] = board[x, y]
        for k in range(1, self.size):
            if not stop_right:
                state[x, y + k] = board[x, y + k]
                state[x + 1, y + k] = 0 if self.board[x + 1, y + k] == 0 else 1
                state[x - 1, y + k] = 0 if self.board[x - 1, y + k] == 0 else 1
                if self.board[x, y + k] == 0:
                    stop_right = True
            if not stop_left:
                state[x, y - k] = board[x, y - k]
                state[x + 1, y - k] = 0 if self.board[x + 1, y - k] == 0 else 1
                state[x - 1, y - k] = 0 if self.board[x - 1, y - k] == 0 else 1
                if self.board[x, y - k] == 0:
                    stop_left = True
            if not stop_top:
                state[x - k, y] = board[x - k, y]
                state[x - k, y + 1] = 0 if self.board[x - k, y + 1] == 0 else 1
                state[x - k, y - 1] = 0 if self.board[x - k, y - 1] == 0 else 1
                if self.board[x - k, y] == 0:
                    stop_top = True
            if not stop_bottom:
                state[x + k, y] = board[x + k, y]
                state[x + k, y + 1] = 0 if self.board[x + k, y + 1] == 0 else 1
                state[x + k, y - 1] = 0 if self.board[x + k, y - 1] == 0 else 1
                if self.board[x + k, y] == 0:
                    stop_bottom = True
        return state

    def _get_state_progressive(self, board, blank):
        x, y = self.agent_position
        state = np.zeros_like(board)
        if blank:  # Useful in the beginning when there are no frames in the memory
            return state
        state[x, y] = board[x, y]
        depth_of_field = config().sim.env.state.depth_of_field
        positions = [self.agent_position]
        recursive_walk(positions, state, board, depth_of_field, self.mask_board, self.seen_positions)
        if config().sim.env.state.type == "memory":
            # we add previously seen positions
            state[self.mask_board] = board[self.mask_board]
        return state

    def _get_state_simple(self, board, blank):
        x, y = self.agent_position
        stop_right = False
        stop_left = False
        stop_top = False
        stop_bottom = False
        state = np.zeros(8)
        if blank:
            return state
        for k in range(1, self.size):
            if not stop_right:
                if board[x, y + k] == 0.3:
                    state[0] += 1
                if self.board[x, y + k] == 0:
                    if k == 1:  # If direct wall, -1 state, otherwise just 0, but we can still go in this direction.
                        state[4] = k
                    stop_right = True
            if not stop_left:
                if board[x, y - k] == 0.3:
                    state[1] += 1
                if self.board[x, y - k] == 0:
                    if k == 1:
                        state[5] = k
                    stop_left = True
            if not stop_top:
                if board[x - k, y] == 0.3:
                    state[2] += 1
                if self.board[x - k, y] == 0:
                    if k == 1:
                        state[6] = k
                    stop_top = True
            if not stop_bottom:
                if board[x + k, y] == 0.3:
                    state[3] += 1
                if self.board[x + k, y] == 0:
                    if k == 1:
                        state[7] = k
                    stop_bottom = True
        return state

    def _get_state(self, board, blank=False):
        if config().sim.env.state.type in ["progressive", "memory"]:
            return self._get_state_progressive(board, blank)
        elif config().sim.env.state.type == "simple":
            return self._get_state_simple(board, blank)
        return self._get_state_hard(board, blank)

    def _get_possible_actions(self):
        x, y = self.agent_position
        possible_actions = []
        if self.board[x, y + 1] == 1:
            possible_actions.append(3)
        if self.board[x, y - 1] == 1:
            possible_actions.append(2)
        if self.board[x + 1, y] == 1:
            possible_actions.append(1)
        if self.board[x - 1, y] == 1:
            possible_actions.append(0)
        return possible_actions

    def _action_to_position(self, action):
        x, y = self.agent_position
        if action == 0:
            return x - 1, y
        if action == 1:
            return x + 1, y
        if action == 2:
            return x, y - 1
        return x, y + 1

    def get_blank_state(self):
        return self._get_state(self._get_board(), blank=True)

    def reset(self):
        self.episode += 1
        self.iter = 1
        self.board_memory = []
        self.seen_positions = []
        self.state_memory = []
        self.board = maze(self.size, self.size)
        self.mask_board = np.full_like(self.board, False, dtype=bool)
        possible_positions = list(zip(*np.where(self.board == 1)))
        self.agent_position = random.sample(possible_positions, 1)[0]
        self.seen_positions.append(self.agent_position)
        self.coin_positions = random.sample(possible_positions, config().sim.env.number_coins)
        board = self._get_board()
        state = self._get_state(board)
        self._add_state(state)
        self._add_board(board)

        return state

    def step(self, action):
        """
        Args:
            action: in ["top", "left", "right", "bottom"]

        Returns: next_state, reward, terminal

        """
        terminal = self.iter == self.max_length
        if action in self._get_possible_actions():  # Si action possible on la fait, sinon on fait rien
            self.agent_position = self._action_to_position(action)
        reward = 0
        if self.agent_position in self.coin_positions:
            reward = 1
            self.coin_positions.remove(self.agent_position)
        board = self._get_board()
        next_state = self._get_state(board)
        self._add_state(next_state)
        self._add_board(board)
        self.iter += 1
        return next_state, reward, terminal

    def make_anim(self, prefix=""):
        for t in range(len(self.board_memory)):
            self.board_memory[t] = cv2.resize(self.board_memory[t], None, fx=self.scale, fy=self.scale,
                                              interpolation=cv2.INTER_NEAREST)
            if config().sim.env.state.type != "simple":
                self.state_memory[t] = cv2.resize(self.state_memory[t], None, fx=self.scale, fy=self.scale,
                                                  interpolation=cv2.INTER_NEAREST)
        filepath = os.path.abspath(os.path.join(config().sim.output.path, f"{prefix}episode-{self.episode}"))
        data = np.array(self.board_memory) * 255
        imageio.mimsave(filepath + '.gif', data.astype(np.uint8), duration=0.1)
        if config().sim.env.state.type != "simple":
            data_state = np.array(self.state_memory) * 255
            imageio.mimsave(filepath + '_state.gif', data_state.astype(np.uint8), duration=0.1)

    def plot(self, state=None):
        assert len(self.board_memory) > 0, "Env has not been reset."
        if state is None:
            state = self.board_memory[-1]
        plt.figure(figsize=(10, 5))
        plt.imshow(state)
        plt.xticks([]), plt.yticks([])
        plt.show()


def recursive_walk(positions, state, board, max_steps, all_positions=None, seen_positions=None):
    x, y = positions[-1]
    if max_steps == 0:
        return
    if board[x + 1, y] != 0 and (x + 1, y) not in positions and (seen_positions is None or not (x + 1, y) in seen_positions):
        positions.append((x + 1, y))
        if all_positions is not None:
            all_positions[positions[-1][0], positions[-1][1]] = True
            seen_positions.append(positions[-1])
        state[x + 1, y] = board[x + 1, y]
        recursive_walk(positions, state, board, max_steps - 1, all_positions, seen_positions)
    if board[x - 1, y] != 0 and (x - 1, y) not in positions and (seen_positions is None or not (x - 1, y) in seen_positions):
        positions.append((x - 1, y))
        if all_positions is not None:
            all_positions[positions[-1][0], positions[-1][1]] = True
            seen_positions.append(positions[-1])
        state[x - 1, y] = board[x - 1, y]
        recursive_walk(positions, state, board, max_steps - 1, all_positions, seen_positions)
    if board[x, y + 1] != 0 and (x, y + 1) not in positions and (seen_positions is None or not (x, y + 1) in seen_positions):
        positions.append((x, y + 1))
        if all_positions is not None:
            all_positions[positions[-1][0], positions[-1][1]] = True
            seen_positions.append(positions[-1])
        state[x, y + 1] = board[x, y + 1]
        recursive_walk(positions, state, board, max_steps - 1, all_positions, seen_positions)
    if board[x, y - 1] != 0 and (x, y - 1) not in positions and (seen_positions is None or not (x, y - 1) in seen_positions):
        positions.append((x, y - 1))
        if all_positions is not None:
            all_positions[positions[-1][0], positions[-1][1]] = True
            seen_positions.append(positions[-1])
        state[x, y - 1] = board[x, y - 1]
        recursive_walk(positions, state, board, max_steps - 1, all_positions, seen_positions)

