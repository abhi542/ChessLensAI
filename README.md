# Chess Scoresheet OCR & Digitizer

A powerful tool that converts handwritten chess scoresheets into digital PGN files using AI-powered OCR. This project creates a bridge between physical chess games and digital analysis tools.

## Key Features

- **AI-Powered OCR**: Uses Groq (Llama Vision) to extract moves from handwritten scoresheets.
- **Interactive Web Interface**: A modern UI to review and edit extracted moves.
- **Real-time Validation**: Validates moves against legal chess rules as you edit.
- **Visual Feedback**: Shows the board state for every move; highlights illegal moves and lets you see the board position *before* the error occurred.
- **PGN Export**: Allows downloading the game as a `.pgn` file once the game record is fully valid.

## Project Structure

- `server.py`: FastAPI backend that handles image processing and game validation.
- `main.py`: Core logic for OCR extraction (Groq) and chess validation (`python-chess`).
- `static/`: Frontend assets (HTML, CSS, JS) for the interactive editor.

## Setup & Installation

### Prerequisites

- Python 3.10+
- A [Groq API Key](https://console.groq.com/keys) (for the Vision LLM).

### 1. Clone & Setup Environment

```bash
# Clone the repository
git clone https://github.com/abhi542/ChessLensAI.git
cd ChessLensAI

# Create a virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration

Create a `.env` file in the root directory and add your Groq API Key:

```ini
GROQ_API_KEY=gsk_...your_key_here...
```

## Running the Application

1.  **Start the Server**:
    ```bash
    uvicorn server:app --reload
    ```
    *The server will start at `http://127.0.0.1:8000`.*

2.  **Open the Web Interface**:
    Navigate to **[http://127.0.0.1:8000/static/index.html](http://127.0.0.1:8000/static/index.html)** in your browser.

## Usage Guide

1.  **Upload**: Click "Upload Image" and select a clear photo of a chess scoresheet.
2.  **Review**: The moves will appear in the grid on the left. Calculated board positions appear on the right.
3.  **Edit**: 
    - If a move is red (invalid), click it to edit.
    - The board will show the position immediately *before* that move, so you can decipher what was played.
    - Correct the text (e.g., change `Nf5` to `Ng5`).
4.  **Export**: Once all moves are green (valid), the "Export PGN" button will enable. Click it to save your game.

## Tech Stack

- **Backend**: Python, FastAPI, python-chess, LangChain (Groq)
- **Frontend**: HTML5, TailwindCSS, jQuery, Chessboard.js
- **Model**: Llama 3.2 (via Groq) for Vision/OCR
