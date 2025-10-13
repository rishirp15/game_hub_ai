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
@app.route('/api/get_ai_move', methods=['POST'])
def get_ai_move():
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
