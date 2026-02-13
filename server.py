import os
import shutil
from typing import List, Optional
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import chess

from main import extract_moves, validate_moves, build_pgn

app = FastAPI()

# Mount static files (frontend)
app.mount("/static", StaticFiles(directory="static"), name="static")

# CORS (allow all for simplicity in local dev)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class MoveRequest(BaseModel):
    move_number: int
    white: Optional[str] = None
    black: Optional[str] = None

class ValidationRequest(BaseModel):
    moves: List[MoveRequest]
    white_player: str = "?"
    black_player: str = "?"
    event: str = "Chess OCR"
    site: str = "?"
    date: Optional[str] = None
    round: str = "?"
    result: str = "*"

@app.get("/")
def read_root():
    return {"message": "Chess OCR API. Go to /static/index.html to use the UI."}

@app.post("/api/upload")
async def upload_image(file: UploadFile = File(...)):
    """
    Upload an image, run OCR, and return the raw extracted moves.
    """
    temp_dir = Path("temp_uploads")
    temp_dir.mkdir(exist_ok=True)
    
    file_path = temp_dir / file.filename
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        print(f"Processing image: {file_path}")
        raw_moves = extract_moves(str(file_path))
        
        # Cleanup
        os.remove(file_path)
        
        return {"moves": raw_moves}
    except Exception as e:
        print(f"Error processing image: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/validate")
async def validate(request: ValidationRequest):
    """
    Validate a sequence of moves.
    Returns:
    - annotated_moves: List of moves with validation status and FENs.
    - valid: Boolean, true if ALL moves are valid.
    - pgn: Generated PGN string (if valid).
    """
    # Convert Pydantic models to dicts for our existing logic
    raw_moves = [m.dict() for m in request.moves]
    
    annotated_moves, board = validate_moves(raw_moves)
    
    # Check if completely valid
    is_valid = all(
        (row["white"] is None or row["white"]["valid"]) and 
        (row["black"] is None or row["black"]["valid"])
        for row in annotated_moves
    )
    
    pgn_string = None
    if is_valid:
        # Generate PGN in memory
        # We need a temporary way to get the string without writing to file?
        # Actually existing build_pgn writes to file. Let's adapt it or just use a temp file.
        # For now, let's use a temp output path
        output_path = f"output/temp_export.pgn"
        pgn_string = build_pgn(
            annotated_moves, 
            board, 
            output_path, 
            white=request.white_player, 
            black=request.black_player, 
            event=request.event,
            site=request.site,
            date_str=request.date,
            round_str=request.round,
            result_str=request.result
        )
        # Read it back (hacky but reuses existing logic)
        with open(output_path, "r") as f:
            pgn_string = f.read()

    return {
        "annotated_moves": annotated_moves,
        "valid": is_valid,
        "pgn": pgn_string
    }

@app.post("/api/upload-pgn")
async def upload_pgn(file: UploadFile = File(...)):
    """
    Upload a PGN file, parse it, and return validated moves.
    """
    temp_dir = Path("temp_uploads")
    temp_dir.mkdir(exist_ok=True)
    file_path = temp_dir / file.filename

    try:
        # Save uploaded file
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Parse PGN
        with open(file_path, "r") as f:
            game = chess.pgn.read_game(f)

        if game is None:
            raise HTTPException(status_code=400, detail="Invalid PGN file")

        # Extract headers
        headers = game.headers
        white_player = headers.get("White", "?")
        black_player = headers.get("Black", "?")
        event = headers.get("Event", "?")
        site = headers.get("Site", "?")
        date_str = headers.get("Date", "")
        round_str = headers.get("Round", "?")
        result = headers.get("Result", "*")

        # Extract moves from mainline
        moves_list = []
        node = game
        move_number = 0
        
        while node.variations:
            next_node = node.variation(0)
            move = next_node.move
            san = node.board().san(move)
            
            # If it's White's turn (board matches node.board()), move_number increments
            if node.board().turn == chess.WHITE:
                move_number += 1
                # Start new row
                moves_list.append({
                    "move_number": move_number,
                    "white": san,
                    "black": None # Will fill if next move exists
                })
            else:
                # Black's move, append to last row
                if moves_list:
                    moves_list[-1]["black"] = san
                else:
                    # Should not happen in standard game starting with black, but handle edge case
                    moves_list.append({
                        "move_number": move_number, # Might be 0 or 1?
                        "white": None,
                        "black": san
                    })
            
            node = next_node

        # Clean nulls (ensure they are structurally compatible with our validator)
        # Our validator expects dicts not strings for raw input? 
        # No, extract_moves returns dicts with "white": "e4". 
        # But validate_moves handles standard dicts.
        
        # Now validate them
        annotated_moves, board = validate_moves(moves_list)
        
        # Cleanup
        os.remove(file_path)

        return {
            "annotated_moves": annotated_moves,
            "valid": True, # PGNs parsed by python-chess are generally valid unless illegal moves forced
            "pgn": str(game), # Return original PGN text? Or re-generated?
            "white_player": white_player,
            "black_player": black_player,
            "event": event,
            "site": site,
            "date": date_str,
            "round": round_str,
            "result": result
        }

    except Exception as e:
        print(f"Error processing PGN: {e}")
        # Cleanup if exists
        if file_path.exists():
            os.remove(file_path)
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
