import os
import requests

pieces = {
    "bR": "https://images.chesscomfiles.com/chess-themes/pieces/neo/150/br.png",
    "bN": "https://images.chesscomfiles.com/chess-themes/pieces/neo/150/bn.png",
    "bB": "https://images.chesscomfiles.com/chess-themes/pieces/neo/150/bb.png",
    "bQ": "https://images.chesscomfiles.com/chess-themes/pieces/neo/150/bq.png",
    "bK": "https://images.chesscomfiles.com/chess-themes/pieces/neo/150/bk.png",
    "bP": "https://images.chesscomfiles.com/chess-themes/pieces/neo/150/bp.png",
    "wR": "https://images.chesscomfiles.com/chess-themes/pieces/neo/150/wr.png",
    "wN": "https://images.chesscomfiles.com/chess-themes/pieces/neo/150/wn.png",
    "wB": "https://images.chesscomfiles.com/chess-themes/pieces/neo/150/wb.png",
    "wQ": "https://images.chesscomfiles.com/chess-themes/pieces/neo/150/wq.png",
    "wK": "https://images.chesscomfiles.com/chess-themes/pieces/neo/150/wk.png",
    "wP": "https://images.chesscomfiles.com/chess-themes/pieces/neo/150/wp.png"
}

output_dir = "static/pieces/neo"
os.makedirs(output_dir, exist_ok=True)

for name, url in pieces.items():
    print(f"Downloading {name}...")
    try:
        r = requests.get(url)
        r.raise_for_status()
        with open(f"{output_dir}/{name}.png", "wb") as f:
            f.write(r.content)
    except Exception as e:
        print(f"Failed to download {name}: {e}")

print("Download complete.")
