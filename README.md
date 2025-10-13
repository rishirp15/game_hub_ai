### 🎮 AI Game Hub

An interactive, web-based platform showcasing three classic strategy games powered by distinct and powerful AI algorithms. This project provides a hands-on experience with Minimax, Alpha-Beta Pruning, and Q-Learning through engaging, responsive gameplay.

-----

### ✨ Key Features

-----

  * 🎲 **Three Classic Games:** Play Connect 4, Othello (Reversi), and Nim.
  * 🤖 **Diverse AI Opponents:** Challenge three unique AI agents, each built with a different algorithm.
  * 🕹️ **Multiple Game Modes:**
      * Human vs. AI
      * Human vs. Human
      * AI vs. AI Simulation
  * 📶 **Variable Difficulty:** Adjust the AI's skill level (Easy, Medium, Hard).
  * 🎨 **Modern & Responsive UI:** A clean, intuitive, and mobile-friendly interface.

-----

### 🎯 Featured Games & AI Algorithms

-----

**🔗 Connect 4**

  * **AI Algorithm:** Minimax with Alpha-Beta Pruning
  * **Description:** A classic search algorithm that finds the optimal move by exploring a limited-depth game tree efficiently.

**⚫ Othello (Reversi)**

  * **AI Algorithm:** Advanced Heuristic Minimax
  * **Description:** A sophisticated Minimax AI that uses positional weights and mobility analysis to make strategic decisions.

**🥢 Nim (Sticks & Stones)**

  * **AI Algorithm:** Q-Learning (Reinforcement Learning)
  * **Description:** An AI that learns the unbeatable mathematical strategy from scratch by playing thousands of simulated games.

-----

### 🛠️ Technology Stack

-----

  * **🖥️ Frontend:** HTML5, CSS3, Tailwind CSS, JavaScript (ES6+)
  * **🐍 Backend:** Python 3, Flask
  * **🧠 AI Algorithms:** Implemented from scratch with no external ML libraries.

-----

### 🚀 Getting Started

-----

Follow these steps to get the project running on your local machine.

**📋 Prerequisites:**

  * Python 3.8 or newer.
  * `pip` for installing Python packages.
  * A modern web browser (Chrome, Firefox, Edge, etc.).

**⚙️ Installation & Setup:**

**1. Clone the Repository**

```
git clone https://github.com/your-username/ai-game-hub.git
cd ai-game-hub
```

**2. Install Python Dependencies**

```
pip install -r requirements.txt
```

**3. Start the Backend Servers**
This project requires **three separate terminals** to run all game servers concurrently.

  * **Terminal 1 (Connect 4):**
    ```
    python connect4_backend.py
    ```
  * **Terminal 2 (Othello/Reversi):**
    ```
    python reversi_backend.py
    ```
  * **Terminal 3 (Nim):**
    ```
    python nim_backend.py
    ```

**4. Launch the Frontend**
With all servers running, open `index.html` in your browser. For the best experience, serve it locally:

  * Open a **fourth terminal** and run:
    ```
    python -m http.server 8000
    ```
  * Navigate to `http://localhost:8000` in your browser.

-----

### 🎮 How to Play

-----

1.  Open the `index.html` file in your browser.
2.  Click "Play Now" to open the game selection modal.
3.  Choose between Connect 4, Othello, or Nim.
4.  Select your desired game mode and AI difficulty.
5.  Enjoy the game\!

-----

### 📁 File Structure

-----

  * 📄 `index.html` - Main frontend file with all UI, CSS, and JS
  * 🐍 `connect4_backend.py` - Flask server & Minimax AI for Connect 4
  * 🐍 `reversi_backend.py` - Flask server & Heuristic AI for Othello
  * 🐍 `nim_backend.py` - Flask server & Q-Learning AI for Nim
  * 📄 `requirements.txt` - Python dependencies
  * 💾 `q_table_nim.json` - Auto-generated file storing the trained Nim AI model
  * 📖 `README.md` - You are here\!

-----

### 🤔 Troubleshooting

-----

  * **"AI Server Error" in game:** Ensure all three Python backend scripts are running in their separate terminals.
  * **Games not loading:**
      * Make sure ports 5000, 5002, and 5003 are not being used by other applications.
      * Check your browser's developer console (F12) for any `CORS` or `Network` errors.
  * **Nim AI slow on first run:** This is expected. The AI is training itself. Subsequent runs will be instant.

-----

Ready to challenge the AI? Fire up the servers and let the games begin\! 🤖