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
@app.route('/api/get_ai_move', methods=['POST'])
def get_ai_move():
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
