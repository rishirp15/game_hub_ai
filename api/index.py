from flask import Flask, request, jsonify
from flask_cors import CORS
import random
import math
import os
import json
import sys

app = Flask(__name__)
CORS(app)

from flask import Flask, request, jsonify
from flask_cors import CORS
import random
import math
import time

app = Flask(__name__)
CORS(app)

# --- Game Constants ---
ROWS = 6
COLS = 7
PLAYER_PIECE = 1
AI_PIECE = 2
EMPTY = 0

def evaluate_window(window, piece):
    """Enhanced window evaluation with better scoring."""
    score = 0
    opponent_piece = PLAYER_PIECE if piece == AI_PIECE else AI_PIECE
    
    piece_count = window.count(piece)
    empty_count = window.count(EMPTY)
    opponent_count = window.count(opponent_piece)
    
    if piece_count == 4:
        score += 100
    elif piece_count == 3 and empty_count == 1:
        score += 10
    elif piece_count == 2 and empty_count == 2:
        score += 2
        
    if opponent_count == 3 and empty_count == 1:
        score -= 80  # Block opponent wins
        
    return score

def score_position(board, piece):
    """Improved position scoring."""
    score = 0
    
    # Center column preference
    center_array = [board[row][COLS//2] for row in range(ROWS)]
    center_count = center_array.count(piece)
    score += center_count * 3
    
    # Horizontal
    for r in range(ROWS):
        for c in range(COLS-3):
            window = [board[r][c+i] for i in range(4)]
            score += evaluate_window(window, piece)
    
    # Vertical
    for c in range(COLS):
        for r in range(ROWS-3):
            window = [board[r+i][c] for i in range(4)]
            score += evaluate_window(window, piece)
    
    # Positive diagonals
    for r in range(ROWS-3):
        for c in range(COLS-3):
            window = [board[r+i][c+i] for i in range(4)]
            score += evaluate_window(window, piece)
    
    # Negative diagonals
    for r in range(ROWS-3):
        for c in range(COLS-3):
            window = [board[r+3-i][c+i] for i in range(4)]
            score += evaluate_window(window, piece)
    
    return score

def check_for_win(board, piece):
    """Check for win condition."""
    # Check horizontal
    for c in range(COLS-3):
        for r in range(ROWS):
            if all(board[r][c+i] == piece for i in range(4)):
                return True
    
    # Check vertical
    for c in range(COLS):
        for r in range(ROWS-3):
            if all(board[r+i][c] == piece for i in range(4)):
                return True
    
    # Check positive diagonal
    for c in range(COLS-3):
        for r in range(ROWS-3):
            if all(board[r+i][c+i] == piece for i in range(4)):
                return True
    
    # Check negative diagonal
    for c in range(COLS-3):
        for r in range(3, ROWS):
            if all(board[r-i][c+i] == piece for i in range(4)):
                return True
    
    return False

def get_valid_locations(board):
    """Get valid moves in center-first order."""
    valid_locations = []
    # Check center first, then alternate outward
    col_order = [3, 2, 4, 1, 5, 0, 6]
    for col in col_order:
        if board[ROWS-1][col] == EMPTY:
            valid_locations.append(col)
    return valid_locations

def get_next_open_row(board, col):
    """Find the next open row in a column."""
    for r in range(ROWS):
        if board[r][col] == EMPTY:
            return r
    return None

def is_terminal_node(board):
    """Check if the game is over."""
    return (check_for_win(board, PLAYER_PIECE) or 
            check_for_win(board, AI_PIECE) or 
            len(get_valid_locations(board)) == 0)

def minimax(board, depth, alpha, beta, maximizing_player):
    """Improved minimax with better evaluation."""
    valid_locations = get_valid_locations(board)
    is_terminal = is_terminal_node(board)
    
    if depth == 0 or is_terminal:
        if is_terminal:
            if check_for_win(board, AI_PIECE):
                return (None, 100000000)
            elif check_for_win(board, PLAYER_PIECE):
                return (None, -10000000)
            else:  # Game is over, no more valid moves
                return (None, 0)
        else:  # Depth is zero
            return (None, score_position(board, AI_PIECE))
    
    if maximizing_player:
        value = -math.inf
        column = random.choice(valid_locations) if valid_locations else None
        for col in valid_locations:
            row = get_next_open_row(board, col)
            if row is not None:
                b_copy = [row[:] for row in board]
                b_copy[row][col] = AI_PIECE
                new_score = minimax(b_copy, depth-1, alpha, beta, False)[1]
                if new_score > value:
                    value = new_score
                    column = col
                alpha = max(alpha, value)
                if alpha >= beta:
                    break
        return column, value
    
    else:  # Minimizing player
        value = math.inf
        column = random.choice(valid_locations) if valid_locations else None
        for col in valid_locations:
            row = get_next_open_row(board, col)
            if row is not None:
                b_copy = [row[:] for row in board]
                b_copy[row][col] = PLAYER_PIECE
                new_score = minimax(b_copy, depth-1, alpha, beta, True)[1]
                if new_score < value:
                    value = new_score
                    column = col
                beta = min(beta, value)
                if alpha >= beta:
                    break
        return column, value

# =============================================================================
# API ENDPOINTS
# =============================================================================

@app.route('/api/connect4/get_ai_move', methods=['POST'])
def c4_get_ai_move():
    try:
        data = request.get_json()
        board = data['board']
        difficulty = data.get('difficulty', 'medium')
        
        depth_map = {'easy': 2, 'medium': 4, 'hard': 6}
        depth = depth_map.get(difficulty.lower(), 4)
        
        col, _ = minimax(board, depth, -math.inf, math.inf, True)
        return jsonify({'move': col})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/ai_vs_ai_simulation', methods=['POST'])
def c4_ai_vs_ai_simulation():
    """Fixed AI vs AI simulation to prevent draws."""
    try:
        board = [[EMPTY for _ in range(COLS)] for _ in range(ROWS)]
        moves_history = []
        current_player = PLAYER_PIECE  # Start with player 1
        move_count = 0
        max_moves = 42  # Maximum possible moves
        
        while not is_terminal_node(board) and move_count < max_moves:
            # Use different depths for variety
            if current_player == PLAYER_PIECE:
                depth = 3  # AI 1 (Yellow) - slightly weaker
                col, _ = minimax(board, depth, -math.inf, math.inf, False)  # Minimizing for player 1
            else:
                depth = 4  # AI 2 (Red) - slightly stronger  
                col, _ = minimax(board, depth, -math.inf, math.inf, True)   # Maximizing for AI
            
            if col is not None:
                row = get_next_open_row(board, col)
                if row is not None:
                    board[row][col] = current_player
                    moves_history.append({
                        'col': col,
                        'row': row, 
                        'player': current_player
                    })
                    move_count += 1
                    
                    # Check for win after move
                    if check_for_win(board, current_player):
                        break
            else:
                break  # No valid moves
            
            # Switch players
            current_player = AI_PIECE if current_player == PLAYER_PIECE else PLAYER_PIECE
        
        # Determine winner
        winner = "Draw"
        if check_for_win(board, PLAYER_PIECE):
            winner = "AI 1 (Yellow)"
        elif check_for_win(board, AI_PIECE):
            winner = "AI 2 (Red)"
        
        return jsonify({
            'moves': moves_history,
            'winner': winner
        })
        
    except Exception as e:
        return jsonify({'error': f'Simulation failed: {str(e)}'}), 500

if __name__ == '__main__':
    print("Starting Connect 4 AI Server...")
    app.run(debug=True, port=5000)


from flask import Flask, request, jsonify
from flask_cors import CORS
import random
import math
import time

app = Flask(__name__)
CORS(app)

# --- Game Constants ---
BOARD_SIZE = 8
BLACK, WHITE, EMPTY = 1, 2, 0

POSITIONAL_WEIGHTS = [
    [120, -20, 20,  5,  5, 20, -20, 120],
    [-20, -40, -5, -5, -5, -5, -40, -20],
    [ 20,  -5, 15,  3,  3, 15,  -5,  20],
    [  5,  -5,  3,  3,  3,  3,  -5,   5],
    [  5,  -5,  3,  3,  3,  3,  -5,   5],
    [ 20,  -5, 15,  3,  3, 15,  -5,  20],
    [-20, -40, -5, -5, -5, -5, -40, -20],
    [120, -20, 20,  5,  5, 20, -20, 120]
]

class ReversiGame:
    """Fixed Reversi game with proper board state management."""
    def __init__(self, board=None):
        if board:
            self.board = [row[:] for row in board]
        else:
            self.board = [[EMPTY for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]
            # Correct starting position
            self.board[3][3], self.board[4][4] = WHITE, WHITE
            self.board[3][4], self.board[4][3] = BLACK, BLACK
            
    def get_tiles_to_flip(self, r_start, c_start, piece):
        """Get all tiles that would be flipped by placing a piece."""
        if not (0 <= r_start < BOARD_SIZE and 0 <= c_start < BOARD_SIZE):
            return []
        
        if self.board[r_start][c_start] != EMPTY:
            return []
        
        opponent = BLACK if piece == WHITE else WHITE
        tiles_to_flip = []
        
        # Check all 8 directions
        directions = [[0, 1], [1, 1], [1, 0], [1, -1], [0, -1], [-1, -1], [-1, 0], [-1, 1]]
        
        for r_dir, c_dir in directions:
            line = []
            r, c = r_start + r_dir, c_start + c_dir
            
            # Collect opponent pieces in this direction
            while (0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE and 
                   self.board[r][c] == opponent):
                line.append((r, c))
                r, c = r + r_dir, c + c_dir
            
            # Check if line ends with our piece (valid capture)
            if (0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE and 
                self.board[r][c] == piece and len(line) > 0):
                tiles_to_flip.extend(line)
        
        return tiles_to_flip

    def get_valid_moves(self, piece):
        """Get all valid moves for a piece."""
        valid_moves = []
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                if self.board[r][c] == EMPTY:
                    flips = self.get_tiles_to_flip(r, c, piece)
                    if flips:  # Must flip at least one tile
                        valid_moves.append((r, c))
        
        # Sort moves by strategic value (corners first)
        def move_priority(move):
            r, c = move
            if (r, c) in [(0,0), (0,7), (7,0), (7,7)]:  # Corners
                return 0
            elif r == 0 or r == 7 or c == 0 or c == 7:  # Edges
                return 1
            else:
                return 2 + (100 - POSITIONAL_WEIGHTS[r][c])  # Interior by position value
        
        return sorted(valid_moves, key=move_priority)

    def make_move(self, r, c, piece):
        """Apply move and return new game state."""
        tiles_to_flip = self.get_tiles_to_flip(r, c, piece)
        if not tiles_to_flip:  # Invalid move
            return None
        
        # Create new board state
        new_board = [row[:] for row in self.board]
        new_board[r][c] = piece
        
        # Flip captured tiles
        for fr, fc in tiles_to_flip:
            new_board[fr][fc] = piece
        
        return ReversiGame(board=new_board)

    def is_game_over(self):
        """Check if game is over (no valid moves for either player)."""
        return (len(self.get_valid_moves(BLACK)) == 0 and 
                len(self.get_valid_moves(WHITE)) == 0)

    def get_score(self):
        """Get current piece count."""
        black_count = sum(row.count(BLACK) for row in self.board)
        white_count = sum(row.count(WHITE) for row in self.board)
        return {BLACK: black_count, WHITE: white_count}

    def print_board(self):
        """Debug method to print board state."""
        print("  0 1 2 3 4 5 6 7")
        for i, row in enumerate(self.board):
            print(f"{i} {' '.join('B' if cell == BLACK else 'W' if cell == WHITE else '.' for cell in row)}")
        print()

class AI:
    """Enhanced AI with better move selection."""
    def __init__(self, piece, difficulty='medium'):
        self.piece = piece
        self.opponent_piece = BLACK if piece == WHITE else WHITE
        self.depth_map = {'easy': 2, 'medium': 4, 'hard': 5}
        self.depth = self.depth_map.get(difficulty.lower(), 4)

    def get_best_move(self, game):
        """Get best move using minimax with optimizations."""
        valid_moves = game.get_valid_moves(self.piece)
        if not valid_moves:
            return None
        
        if len(valid_moves) == 1:
            return valid_moves[0]
        
        # Quick win/corner check
        for r, c in valid_moves:
            if (r, c) in [(0,0), (0,7), (7,0), (7,7)]:  # Take corner immediately
                return (r, c)
        
        move, _ = self._minimax(game, self.depth, -math.inf, math.inf, True)
        return move if move else valid_moves[0]

    def _score_position(self, game):
        """Enhanced position evaluation."""
        if game.is_game_over():
            final_score = game.get_score()
            diff = final_score[self.piece] - final_score[self.opponent_piece]
            if diff > 0:
                return 10000 + diff
            elif diff < 0:
                return -10000 + diff
            else:
                return 0

        # Calculate scores
        my_pieces = opponent_pieces = 0
        positional_score = 0
        
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                if game.board[r][c] == self.piece:
                    positional_score += POSITIONAL_WEIGHTS[r][c]
                    my_pieces += 1
                elif game.board[r][c] == self.opponent_piece:
                    positional_score -= POSITIONAL_WEIGHTS[r][c]
                    opponent_pieces += 1
        
        total_pieces = my_pieces + opponent_pieces
        
        # Endgame: focus on piece count
        if total_pieces > 50:
            return (my_pieces - opponent_pieces) * 50 + positional_score * 0.5
        
        # Mid-game: balance position and mobility
        my_moves = len(game.get_valid_moves(self.piece))
        opponent_moves = len(game.get_valid_moves(self.opponent_piece))
        
        mobility_score = 0
        if my_moves + opponent_moves > 0:
            mobility_score = 50 * (my_moves - opponent_moves) / (my_moves + opponent_moves)
        
        return positional_score + mobility_score

    def _minimax(self, game, depth, alpha, beta, is_maximizing_player):
        """Minimax with alpha-beta pruning."""
        if depth == 0 or game.is_game_over():
            return (None, self._score_position(game))

        current_player_piece = self.piece if is_maximizing_player else self.opponent_piece
        valid_moves = game.get_valid_moves(current_player_piece)
        
        if not valid_moves:
            # Forced pass - switch players
            return self._minimax(game, depth, alpha, beta, not is_maximizing_player)

        best_move = valid_moves[0]
        
        if is_maximizing_player:
            max_eval = -math.inf
            for r, c in valid_moves:
                new_game_state = game.make_move(r, c, current_player_piece)
                if new_game_state:
                    _, eval_score = self._minimax(new_game_state, depth - 1, alpha, beta, False)
                    if eval_score > max_eval:
                        max_eval = eval_score
                        best_move = (r, c)
                    alpha = max(alpha, eval_score)
                    if beta <= alpha:
                        break
            return best_move, max_eval
        else:
            min_eval = math.inf
            for r, c in valid_moves:
                new_game_state = game.make_move(r, c, current_player_piece)
                if new_game_state:
                    _, eval_score = self._minimax(new_game_state, depth - 1, alpha, beta, True)
                    if eval_score < min_eval:
                        min_eval = eval_score
                        best_move = (r, c)
                    beta = min(beta, eval_score)
                    if beta <= alpha:
                        break
            return best_move, min_eval

# --- API Endpoints ---
@app.route('/api/reversi/get_ai_move', methods=['POST'])
def reversi_get_ai_move():
    try:
        data = request.get_json()
        game = ReversiGame(board=data['board'])
        ai = AI(WHITE, difficulty=data.get('difficulty', 'medium'))
        move = ai.get_best_move(game)
        return jsonify({'move': move})
    except Exception as e:
        print(f"AI move error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/validate_move', methods=['POST'])
def validate_move():
    try:
        data = request.get_json()
        game = ReversiGame(board=data['board'])
        r, c = data['move'][0], data['move'][1]
        player = data['player']
        flipped = game.get_tiles_to_flip(r, c, player)
        return jsonify({
            'is_valid': len(flipped) > 0,
            'flipped': flipped
        })
    except Exception as e:
        print(f"Validation error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/get_valid_moves', methods=['POST'])
def api_get_valid_moves():
    try:
        data = request.get_json()
        game = ReversiGame(board=data['board'])
        moves = game.get_valid_moves(data['player'])
        return jsonify({'moves': moves})
    except Exception as e:
        print(f"Valid moves error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/ai_vs_ai_simulation', methods=['POST'])
def ai_vs_ai_simulation():
    """Fixed AI vs AI simulation with proper game progression."""
    try:
        game = ReversiGame()  # Start with standard setup
        ai_black = AI(BLACK, 'medium')
        ai_white = AI(WHITE, 'medium')
        moves_history = []
        current_player = BLACK
        consecutive_passes = 0
        move_count = 0
        max_moves = 60  # Safety limit

        while not game.is_game_over() and move_count < max_moves and consecutive_passes < 2:
            valid_moves = game.get_valid_moves(current_player)
            
            if not valid_moves:
                consecutive_passes += 1
                current_player = WHITE if current_player == BLACK else BLACK
                continue
            
            consecutive_passes = 0
            
            ai = ai_black if current_player == BLACK else ai_white
            move = ai.get_best_move(game)
            
            if move:
                r, c = move
                flipped = game.get_tiles_to_flip(r, c, current_player)
                new_game = game.make_move(r, c, current_player)
                
                if new_game:
                    game = new_game
                    moves_history.append({
                        'row': r, 
                        'col': c, 
                        'player': current_player,
                        'flipped': flipped, 
                        'board_after': [row[:] for row in game.board] # Deep copy
                    })
                    move_count += 1
                else:
                    # AI returned an invalid move, break the loop
                    break
            else:
                # AI failed to return a move, break
                break
            
            current_player = WHITE if current_player == BLACK else BLACK
        
        # Determine winner
        final_score = game.get_score()
        winner = "Draw"
        if final_score[BLACK] > final_score[WHITE]:
            winner = "AI 1 (Black)"
        elif final_score[WHITE] > final_score[BLACK]:
            winner = "AI 2 (White)"
        
        return jsonify({
            'moves': moves_history,
            'winner': winner,
            'final_score': final_score,
            'total_moves': len(moves_history)
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Simulation failed: {str(e)}'}), 500

if __name__ == '__main__':
    print("Starting Fixed Othello (Reversi) AI Server...")
    app.run(debug=True, port=5002)

import os
import json
import random
import sys
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# --- Game & AI Constants ---
INITIAL_STICKS = 21
Q_TABLE_FILE = 'q_table_nim.json'

class QLearningAgent:
    """Fixed Q-Learning agent with proper game ending."""
    def __init__(self, alpha=0.1, gamma=0.9, epsilon=0.1):
        self.q_table = {}
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

    def get_valid_actions(self, sticks):
        if sticks <= 0:
            return []
        return [1, 2, 3] if sticks > 3 else list(range(1, sticks + 1))

    def get_q_value(self, state, action):
        return self.q_table.get(str(state), {}).get(str(action), 0.0)

    def choose_action(self, sticks, epsilon=None):
        if epsilon is None:
            epsilon = self.epsilon
        
        valid_actions = self.get_valid_actions(sticks)
        if not valid_actions:
            return None

        if random.uniform(0, 1) < epsilon:
            return random.choice(valid_actions)

        q_values = {action: self.get_q_value(sticks, action) for action in valid_actions}
        max_q = max(q_values.values())
        best_actions = [action for action, q in q_values.items() if q == max_q]
        return random.choice(best_actions)

    def learn(self, state, action, reward, next_state):
        old_value = self.get_q_value(state, action)
        
        next_valid_actions = self.get_valid_actions(next_state)
        next_max = max([self.get_q_value(next_state, act) for act in next_valid_actions]) if next_valid_actions else 0
        
        new_value = old_value + self.alpha * (reward + self.gamma * next_max - old_value)
        
        state_key = str(state)
        if state_key not in self.q_table:
            self.q_table[state_key] = {}
        self.q_table[state_key][str(action)] = new_value

def train_agent(episodes=100000):
    agent = QLearningAgent()
    print("Training Q-Learning agent for Nim...")
    
    for i in range(episodes):
        sticks = INITIAL_STICKS
        agent_turn = random.choice([True, False])
        
        # Store agent moves for learning
        agent_moves = []
        
        while sticks > 0:
            if agent_turn:
                action = agent.choose_action(sticks)
                if action and action <= sticks:
                    agent_moves.append((sticks, action))
                    sticks -= action
                    
                    if sticks == 0:  # Agent loses (took last stick)
                        for state, act in agent_moves:
                            agent.learn(state, act, -10, 0)
                        break
            else:
                # Opponent plays randomly
                valid_actions = agent.get_valid_actions(sticks)
                if valid_actions:
                    opponent_action = random.choice(valid_actions)
                    sticks -= opponent_action
                    
                    if sticks == 0:  # Opponent loses, agent wins
                        for state, act in agent_moves:
                            agent.learn(state, act, 10, 0)
                        break
            
            agent_turn = not agent_turn
        
        if (i + 1) % 10000 == 0:
            progress = (i + 1) / episodes * 100
            sys.stdout.write(f"\rTraining Progress: {progress:.0f}%")
            sys.stdout.flush()
    
    print("\nTraining complete.")
    agent.epsilon = 0
    return agent.q_table

def load_or_train_q_table():
    if os.path.exists(Q_TABLE_FILE):
        print("Loading pre-trained Q-table for Nim...")
        with open(Q_TABLE_FILE, 'r') as f:
            return json.load(f)
    else:
        print("Q-table not found, starting training...")
        q_table = train_agent()
        with open(Q_TABLE_FILE, 'w') as f:
            json.dump(q_table, f)
        return q_table

q_table_global = load_or_train_q_table()
print("Nim AI is ready.")

# --- API Endpoints ---
@app.route('/api/nim/get_ai_move', methods=['POST'])
def nim_get_ai_move():
    try:
        data = request.get_json()
        if not data or 'sticks' not in data:
            return jsonify({'error': "Request must include 'sticks' count."}), 400
        
        sticks = data['sticks']
        difficulty = data.get('difficulty', 'medium')
        epsilon_map = {'easy': 0.5, 'medium': 0.1, 'hard': 0.0}
        
        agent = QLearningAgent()
        agent.q_table = q_table_global
        move = agent.choose_action(sticks, epsilon=epsilon_map.get(difficulty, 0.1))
        
        return jsonify({'move': move})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/validate_move', methods=['POST'])
def validate_move():
    try:
        data = request.get_json()
        if not data or 'sticks' not in data or 'take' not in data:
            return jsonify({'error': "Request must include 'sticks' and 'take'."}), 400
        
        sticks, take = data['sticks'], data['take']
        is_valid = take in [1, 2, 3] and take <= sticks and take > 0
        return jsonify({'is_valid': is_valid})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/ai_vs_ai_simulation', methods=['POST'])
def ai_vs_ai_simulation():
    """Fixed AI vs AI simulation."""
    try:
        sticks = INITIAL_STICKS
        agent = QLearningAgent()
        agent.q_table = q_table_global
        moves_history = []
        
        current_player = "AI 1"
        
        while sticks > 0:
            # Use different strategies for variety
            epsilon = 0.0 if current_player == "AI 1" else 0.05
            move = agent.choose_action(sticks, epsilon=epsilon)
            
            if move and move <= sticks:
                moves_history.append({
                    'player': current_player,
                    'move': move,
                    'remaining': sticks - move
                })
                sticks -= move
                
                # Check if game ended
                if sticks == 0:
                    # Current player loses (took last stick)
                    winner = "AI 2" if current_player == "AI 1" else "AI 1"
                    break
            else:
                # Should not happen, but safety check
                winner = "AI 2" if current_player == "AI 1" else "AI 1"
                break
            
            # Switch players
            current_player = "AI 2" if current_player == "AI 1" else "AI 1"
        
        return jsonify({
            'moves': moves_history,
            'winner': winner
        })
        
    except Exception as e:
        return jsonify({'error': f'Simulation failed: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5003)
