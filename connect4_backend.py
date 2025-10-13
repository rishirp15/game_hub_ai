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

@app.route('/api/get_ai_move', methods=['POST'])
def get_ai_move():
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
def ai_vs_ai_simulation():
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
