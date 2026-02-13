
import os
import shutil
import uvicorn
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import chess.pgn

# Modular Imports
import config
import services
from schema import ValidationRequest, ValidationResponse

# Initialize FastAPI
app = FastAPI(
    title="ChessLensAI API",
    description="Backend for ChessLensAI: OCR, PGN parsing, and Validation.",
    version="1.0.0"
)

# ── Middleware ───────────────────────────────────────────────────────────────

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Static Files (Frontend) ──────────────────────────────────────────────────

# Ensure static directory exists
if Path("static").exists():
    app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def read_root():
    return {"message": "ChessLensAI API is running. Visit /static/index.html"}


# ── API Endpoints ────────────────────────────────────────────────────────────

@app.get("/health")
def health_check():
    """Health check endpoint to verify service status."""
    return {"status": "healthy", "model": config.MODEL_NAME}


@app.post("/api/upload")
async def upload_image(file: UploadFile = File(...)):
    """
    1. Upload Image
    2. Run OCR (via LLM Service)
    3. Return Raw Moves
    """
    temp_dir = Path("temp_uploads")
    temp_dir.mkdir(exist_ok=True)
    
    file_path = temp_dir / file.filename
    try:
        # Save temp file
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        print(f"[INFO] Processing image: {file_path}")
        
        # Call Service
        raw_moves = services.extract_moves(str(file_path))
        
        return {"moves": raw_moves}

    except Exception as e:
        print(f"[ERROR] Image processing failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Cleanup
        if file_path.exists():
            os.remove(file_path)


@app.post("/api/validate", response_model=ValidationResponse)
async def validate_game(request: ValidationRequest):
    """
    1. Receive Moves
    2. Validate against Chess Rules (Service)
    3. Return Annotated Moves + PGN
    """
    # Convert Pydantic models to list of dicts for service
    raw_moves = [m.dict() for m in request.moves]
    
    annotated_moves, board = services.validate_moves(raw_moves)
    
    # Check if completely valid
    is_valid = all(
        (row["white"] is None or row["white"]["valid"]) and 
        (row["black"] is None or row["black"]["valid"])
        for row in annotated_moves
    )
    
    pgn_string = None
    if is_valid:
        # Generate PGN (using a temp file as required by current build_pgn interface)
        output_path = config.OUTPUT_DIR / "temp_web_export.pgn"
        pgn_string = services.build_pgn(
            annotated_moves, 
            board, 
            str(output_path), 
            white=request.white_player, 
            black=request.black_player, 
            event=request.event,
            site=request.site,
            date_str=request.date,
            round_str=request.round,
            result_str=request.result
        )
        # We can return the string directly, `build_pgn` returns it too.

    return {
        "annotated_moves": annotated_moves,
        "valid": is_valid,
        "pgn": pgn_string
    }


@app.post("/api/upload-pgn")
async def upload_pgn_file(file: UploadFile = File(...)):
    """
    1. Upload PGN File
    2. Parse PGN
    3. Extract & Validate Moves
    4. Return State
    """
    temp_dir = Path("temp_uploads")
    temp_dir.mkdir(exist_ok=True)
    file_path = temp_dir / file.filename

    try:
        # Save temp file
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Parse PGN using python-chess
        with open(file_path, "r") as f:
            game = chess.pgn.read_game(f)

        if game is None:
            raise HTTPException(status_code=400, detail="Invalid or empty PGN file")

        # Extract Headers
        headers = game.headers
        metadata = {
            "white_player": headers.get("White", "?"),
            "black_player": headers.get("Black", "?"),
            "event": headers.get("Event", "?"),
            "site": headers.get("Site", "?"),
            "date": headers.get("Date", ""),
            "round": headers.get("Round", "?"),
            "result": headers.get("Result", "*"),
        }

        # Extract Moves from Mainline
        moves_list = []
        node = game
        move_number = 0
        
        while node.variations:
            next_node = node.variation(0)
            move = next_node.move
            san = node.board().san(move)
            
            if node.board().turn == chess.WHITE:
                move_number += 1
                moves_list.append({
                    "move_number": move_number,
                    "white": san,
                    "black": None
                })
            else:
                if moves_list:
                    moves_list[-1]["black"] = san
                else:
                    # Rare: Black starts (e.g. from position), handle gracefully
                    moves_list.append({
                        "move_number": move_number,
                        "white": None,
                        "black": san
                    })
            
            node = next_node
        
        # Validate Extracted Moves
        annotated_moves, board = services.validate_moves(moves_list)

        return {
            "annotated_moves": annotated_moves,
            "valid": True, 
            "pgn": str(game),
            **metadata
        }

    except Exception as e:
        print(f"[ERROR] PGN Upload failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if file_path.exists():
            os.remove(file_path)


if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    print(f"Starting ChessLensAI API on port {port}...")
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=True)
