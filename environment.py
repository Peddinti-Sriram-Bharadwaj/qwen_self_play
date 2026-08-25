import numpy as np

class TicTacToeEnv:
    def __init__(self):
        self.board = np.zeros(9, dtype=int)
        self.current_player = 1 # 1 for X, -1 for O
        self.winner = None
        self.done = False

    def reset(self):
        self.board = np.zeros(9, dtype=int)
        self.current_player = 1
        self.winner = None
        self.done = False
        return self._get_obs()

    def _get_obs(self):
        """Returns a string representation of the board for the LLM."""
        chars = {0: '.', 1: 'X', -1: 'O'}
        board_str = ""
        for i in range(3):
            row = [chars[self.board[i*3+j]] for j in range(3)]
            board_str += " | ".join(row) + "\n"
            if i < 2:
                board_str += "--+---+--\n"
        
        # Determine valid moves
        valid_moves = [str(i) for i in range(9) if self.board[i] == 0]
        
        prompt = (
            f"Current board:\n{board_str}\n"
            f"You are playing as {'X' if self.current_player == 1 else 'O'}.\n"
            f"Valid moves are: {', '.join(valid_moves)}.\n"
            f"Choose a move (just the number):"
        )
        return prompt

    def step(self, action: int):
        """
        Executes a move. 
        Returns: (observation, reward, done, info)
        """
        if self.done:
            return self._get_obs(), 0, True, {"msg": "Game already finished."}
            
        if action < 0 or action > 8 or self.board[action] != 0:
            # Invalid move, penalize heavily and end game
            self.done = True
            return self._get_obs(), -10.0, True, {"msg": "Invalid move."}

        # Make move
        self.board[action] = self.current_player
        
        # Check win or draw
        if self._check_win(self.current_player):
            self.winner = self.current_player
            self.done = True
            reward = 1.0 # 1 for winning
        elif self._check_draw():
            self.done = True
            reward = 0.0 # 0 for draw
        else:
            self.done = False
            reward = 0.0 # 0 for intermediate step
            self.current_player *= -1 # switch player
            
        return self._get_obs(), reward, self.done, {"winner": self.winner}

    def _check_win(self, player):
        win_conditions = [
            [0, 1, 2], [3, 4, 5], [6, 7, 8], # rows
            [0, 3, 6], [1, 4, 7], [2, 5, 8], # cols
            [0, 4, 8], [2, 4, 6]             # diagonals
        ]
        return any(all(self.board[i] == player for i in combo) for combo in win_conditions)

    def _check_draw(self):
        return not any(self.board == 0)
