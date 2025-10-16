# 🎮 AI Game Hub

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An interactive, web-based platform designed to demonstrate and compare three distinct Artificial Intelligence paradigms through classic strategy games. This project is pre-configured for easy, one-click deployment on Vercel.

---

### ✨ Key Features

* 🎲 **Three Classic Games:** Play Connect 4, Othello (Reversi), and Nim.
* 🤖 **Three Distinct AI Opponents:** Challenge unique AI agents built with Minimax, Advanced Heuristics, and Q-Learning.
* 🕹️ **Multiple Game Modes:** Choose between Human vs. AI, Human vs. Human, or watch an AI vs. AI simulation.
* 📶 **Variable Difficulty:** Adjust the AI's skill level (Easy, Medium, Hard).
* 🎨 **Modern & Responsive UI:** A clean, intuitive interface built with TailwindCSS.
* ☁️ **Ready for Deployment:** Pre-configured for easy deployment as a serverless application on Vercel.

---

### 🎯 Featured Games & AI Algorithms

| Game                 | AI Algorithm Implemented            | Type of AI                          |
| -------------------- | ----------------------------------- | ----------------------------------- |
| **🔗 Connect 4** | Minimax with Alpha-Beta Pruning     | **Classical Search Algorithm** |
| **⚫ Othello (Reversi)**| Minimax with Advanced Heuristics    | **Expert System / Heuristic-Driven**|
| **🥢 Nim** | Q-Learning                          | **Reinforcement Learning** |

---

### 🛠️ Technology Stack

* **🖥️ Frontend:** HTML5, Tailwind CSS, JavaScript (ES6+)
* **🐍 Backend:** Python 3, Flask
* **☁️ Deployment:** Vercel (as a Serverless Function)
* **🧠 AI Algorithms:** Implemented from scratch with no external ML libraries.

---

### 🚀 Getting Started

You can either run this project on your local machine for development or deploy it directly to the web via Vercel.

#### **Option 1: Deploying to Vercel (Recommended)**

This is the easiest way to get the project running.

1.  **Push to GitHub:** Make sure your entire project, including the `vercel.json` and `requirements.txt` files, is in a GitHub repository.
2.  **Import to Vercel:**
    * Go to your Vercel dashboard and click "Add New..." -> "Project".
    * Import the GitHub repository.
    * Vercel will automatically detect the `vercel.json` configuration. No changes are needed.
3.  **Deploy:** Click the "Deploy" button. Your AI Game Hub will be live on a public URL in minutes!

#### **Option 2: Running Locally**

**📋 Prerequisites:**

* Python 3.8 or newer.
* `pip` for installing Python packages.
* A modern web browser.
* (Recommended) VS Code with the "Live Server" extension.

**⚙️ Installation & Setup:**

1.  **Clone the Repository**
    ```bash
    git clone [https://github.com/your-username/ai-game-hub.git](https://github.com/your-username/ai-game-hub.git)
    cd ai-game-hub
    ```

2.  **Install Python Dependencies**
    It's highly recommended to use a virtual environment.
    ```bash
    # Create and activate a virtual environment (optional but good practice)
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`

    # Install packages
    pip install -r requirements.txt
    ```

3.  **Start the Backend Server**
    Run the single, merged Flask application.
    ```bash
    flask --app api/index run
    ```
    The backend server will start, typically on `http://127.0.0.1:5000`.

4.  **Launch the Frontend**
    Open `index.html` in your browser. For the best experience (to avoid potential CORS issues), use a local server. If you have VS Code, you can right-click `index.html` and select "Open with Live Server".

---

### 📁 File Structure

The project is structured for Vercel's serverless environment.

* 📄 `index.html` - The single-page application containing all UI, CSS, and JS.
* `api/`
    * 🐍 `index.py` - The unified Flask server that contains the logic for all three games.
* 📄 `requirements.txt` - A list of Python dependencies for Vercel to install.
* 📄 `vercel.json` - Configuration file that tells Vercel how to build and route the project.
* 💾 `q_table_nim.json` - The pre-trained "brain" for the Nim AI. It must be present for the Nim game to work instantly.

---

### 🤔 Troubleshooting

* **"AI Server Error" in game:** If running locally, ensure the single `flask --app api/index run` command is active and did not crash. Check the terminal for any Python errors.
* **Games not loading locally:**
    * Make sure port 5000 is not being used by another application.
    * Check your browser's developer console (F12) for any `Network` errors. The frontend must be able to reach `http://127.0.0.1:5000`.
* **Nim AI slow on first run (if `q_table_nim.json` is deleted):** This is expected. The AI is training itself by playing 100,000 games. This only happens once. Subsequent runs will be instant as it will load from the generated file.

---

Ready to challenge the AI? Deploy to Vercel or fire it up locally and let the games begin! 🤖