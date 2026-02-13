// -- Constants --
const API_BASE = "http://localhost:8000"; // Assuming local dev
const START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

// -- State --
let gameState = {
    moves: [], // Array of { move_number, white: {san, valid, fen?}, black: {...} }
    currentMoveIndex: -1, // -1 = start, 0 = 1. White, 1 = 1. Black, etc.
    fens: [START_FEN], // Array of FENs matching the move list (0 = Start)
    isValid: false,
    pgn: ""
};

let board = null;
let game = new Chess(); // Local chess.js instance for move validation/generation

// -- Initialization --
$(document).ready(() => {
    // Initialize Chessboard
    board = Chessboard('board', {
        position: 'start',
        pieceTheme: 'pieces/neo/{piece}.png',
        draggable: true,
        onDragStart: onDragStart,
        onDrop: onDrop,
        onSnapEnd: onSnapEnd
    });

    // Event Listeners
    $('#imageInput').on('change', handleImageUpload);
    $('#pgnInput').on('change', handlePgnUpload);
    $('#exportBtn').on('click', handleExport);
    $('#btnFlip').on('click', () => board.flip());

    // Trigger validation when metadata changes
    $('#whitePlayer, #blackPlayer, #eventName, #siteName, #gameDate, #roundNum, #gameResult').on('change', validateMoves);

    // Board Navigation
    $('#btnStart').on('click', () => goToMove(-1));
    $('#btnPrev').on('click', () => goToMove(gameState.currentMoveIndex - 1));
    $('#btnNext').on('click', () => goToMove(gameState.currentMoveIndex + 1));
    $('#btnEnd').on('click', () => goToMove(gameState.fens.length - 2));

    // Global Key Listener
    $(document).on('keydown', (e) => {
        if (e.key === "ArrowLeft") $('#btnPrev').click();
        if (e.key === "ArrowRight") $('#btnNext').click();
        if (e.key === "f") board.flip(); // Optional hotkey
    });
});


// -- Handlers --

async function handleImageUpload(e) {
    const file = e.target.files[0];
    if (!file) return;

    // Show Loading
    $('#loadingModal').removeClass('hidden');

    const formData = new FormData();
    formData.append("file", file);

    try {
        const res = await fetch(`${API_BASE}/api/upload`, {
            method: 'POST',
            body: formData
        });

        if (!res.ok) throw new Error("Upload failed");

        const data = await res.json();

        // Initial raw moves
        gameState.moves = data.moves;

        // Render and validate immediately
        renderGrid();
        await validateMoves();

    } catch (err) {
        alert("Error uploading image: " + err.message);
    } finally {
        $('#loadingModal').addClass('hidden');
    }
}

async function validateMoves() {
    // Collect moves from the grid
    const movesToSend = [];
    const rows = $('.move-row');

    rows.each((i, row) => {
        const num = $(row).find('.move-num').text().replace('.', '');
        const white = $(row).find('.move-white input').val().trim() || null;
        const black = $(row).find('.move-black input').val().trim() || null;

        movesToSend.push({
            move_number: parseInt(num),
            white: white === "" ? null : white,
            black: black === "" ? null : black
        });
    });

    const payload = {
        moves: movesToSend,
        white_player: $('#whitePlayer').val(),
        black_player: $('#blackPlayer').val(),
        event: $('#eventName').val(),
        site: $('#siteName').val(),
        date: $('#gameDate').val(),
        round: $('#roundNum').val(),
        result: $('#gameResult').val()
    };

    try {
        const res = await fetch(`${API_BASE}/api/validate`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });

        const data = await res.json();

        // Update State
        gameState.isValid = data.valid;
        gameState.pgn = data.pgn;

        // Re-construct FEN list
        gameState.fens = [START_FEN];
        let lastValidFen = START_FEN;

        // Update UI Validation status
        data.annotated_moves.forEach((row, i) => {
            const rowEl = $(`.move-row[data-idx="${i}"]`);

            // White
            updateCellStatus(rowEl.find('.move-white'), row.white);
            if (row.white && row.white.valid) {
                gameState.fens.push(row.white.fen);
                lastValidFen = row.white.fen;
            } else if (row.white) {
                // Push last valid so navigation doesn't break, or push the error FEN if backend sent it (it sends PREVIOUS fen on error)
                gameState.fens.push(row.white.fen || lastValidFen);
            } else {
                gameState.fens.push(lastValidFen); // Null move?
            }

            // Black
            updateCellStatus(rowEl.find('.move-black'), row.black);
            if (row.black && row.black.valid) {
                gameState.fens.push(row.black.fen);
                lastValidFen = row.black.fen;
            } else if (row.black) {
                gameState.fens.push(row.black.fen || lastValidFen);
            } else {
                gameState.fens.push(lastValidFen);
            }
        });

        // Toggle Export Button
        $('#exportBtn').prop('disabled', !gameState.isValid);

        // If we just validated, update board to the "latest" relevant position? 
        // Or keep current? Let's stay current unless out of bounds.
        // Actually best UX: If invalid, jump to the first error? 
        // For now, simple: just refresh view
        updateBoardStatus();

    } catch (err) {
        console.error("Validation error:", err);
    }
}

function updateCellStatus(cell, data) {
    const input = cell.find('input');
    cell.removeClass('bg-green-900/20 bg-red-900/30');
    input.removeClass('text-green-300 text-red-300 line-through');

    if (!data) return; // Empty

    if (data.valid) {
        cell.addClass('bg-green-900/20');
        input.addClass('text-green-300');
    } else {
        cell.addClass('bg-red-900/30');
        input.addClass('text-red-300');
        // cell.attr('title', data.error); // Tooltip
    }
}

function renderGrid() {
    const container = $('#movesGrid');
    container.empty();

    gameState.moves.forEach((row, idx) => {
        const whiteSan = row.white ? (typeof row.white === 'string' ? row.white : row.white.san) : "";
        const blackSan = row.black ? (typeof row.black === 'string' ? row.black : row.black.san) : "";

        const html = `
        <div class="grid grid-cols-[3rem_1fr_1fr] border-b border-gray-700 move-row" data-idx="${idx}">
            <div class="py-2 text-center text-gray-500 font-mono text-sm move-num">${row.move_number}.</div>
            
            <div class="move-cell move-white p-1">
                <input type="text" value="${whiteSan || ''}" 
                    class="move-input w-full h-full bg-transparent text-center focus:outline-none text-gray-200"
                    onchange="validateMoves()"
                    onfocus="highlightMove(${idx}, 'white')">
            </div>
            
            <div class="move-cell move-black p-1 border-l border-gray-700">
                <input type="text" value="${blackSan || ''}" 
                    class="move-input w-full h-full bg-transparent text-center focus:outline-none text-gray-200"
                    onchange="validateMoves()"
                    onfocus="highlightMove(${idx}, 'black')">
            </div>
        </div>
        `;
        container.append(html);
    });
}

// -- Board Interaction --

function goToMove(index) {
    // index is in terms of half-moves (0 = after 1. White, 1 = after 1. Black)
    // -1 = Start Position

    // Bounds check
    if (index < -1) index = -1;
    if (index >= gameState.fens.length - 1) index = gameState.fens.length - 2;

    gameState.currentMoveIndex = index;
    updateBoardStatus();
}

function updateBoardStatus() {
    // The FEN array has Start + valid/invalid states.
    // Index mapping: 
    // -1 -> fens[0] (Start)
    // 0 -> fens[1] (After 1. White)
    // 1 -> fens[2] (After 1. Black)

    const fenIndex = gameState.currentMoveIndex + 1;
    if (fenIndex < 0 || fenIndex >= gameState.fens.length) return;

    const fen = gameState.fens[fenIndex];
    if (fen) {
        board.position(fen);
        game.load(fen);
    }

    // Update active highlight in grid
    $('.move-input').removeClass('ring-2 ring-yellow-500 bg-gray-800 rounded');

    if (gameState.currentMoveIndex >= 0) {
        // Calculate which cell corresponds to this half-move index
        // even index (0, 2...) -> White
        // odd  index (1, 3...) -> Black
        const rowIdx = Math.floor(gameState.currentMoveIndex / 2);
        const isWhite = (gameState.currentMoveIndex % 2) === 0;

        const row = $(`.move-row[data-idx="${rowIdx}"]`);
        const cell = isWhite ? row.find('.move-white input') : row.find('.move-black input');
        cell.addClass('ring-2 ring-yellow-500 bg-gray-800 rounded');

        // Scroll to view
        cell[0].scrollIntoView({ behavior: "smooth", block: "center" });
    }
}

function highlightMove(rowIdx, color) {
    // Convert row+color to half-move index
    // row 0, white -> 0
    // row 0, black -> 1
    // row 1, white -> 2
    let index = rowIdx * 2;
    if (color === 'black') index += 1;

    goToMove(index);
}

function handleExport() {
    if (!gameState.pgn) return;

    const blob = new Blob([gameState.pgn], { type: 'text/plain' });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `game_${Date.now()}.pgn`;
    document.body.appendChild(a);
    a.click();
    window.URL.revokeObjectURL(url);
    document.body.removeChild(a);
}

async function handlePgnUpload(e) {
    const file = e.target.files[0];
    if (!file) return;

    // Show Loading
    $('#loadingModal').removeClass('hidden');

    const formData = new FormData();
    formData.append("file", file);

    try {
        const res = await fetch(`${API_BASE}/api/upload-pgn`, {
            method: 'POST',
            body: formData
        });

        if (!res.ok) throw new Error("Upload failed");

        const data = await res.json();

        // Update Metadata
        $('#whitePlayer').val(data.white_player || "?");
        $('#blackPlayer').val(data.black_player || "?");
        $('#eventName').val(data.event || "?");
        $('#siteName').val(data.site || "?");
        $('#gameDate').val(data.date || "");
        $('#roundNum').val(data.round || "?");
        $('#gameResult').val(data.result || "*");

        // Update Game State
        gameState.isValid = data.valid;
        gameState.pgn = data.pgn;

        // Populate Grid (annotated_moves has same structure as validation output)
        gameState.moves = data.annotated_moves.map(row => ({
            move_number: row.move_number,
            white: row.white && row.white.san ? row.white.san : null,
            black: row.black && row.black.san ? row.black.san : null
        }));

        renderGrid();

        // Re-construct FEN list logic (Shared with validation)
        gameState.fens = [START_FEN];
        let lastValidFen = START_FEN;

        data.annotated_moves.forEach((row, i) => {
            const rowEl = $(`.move-row[data-idx="${i}"]`);

            // White
            updateCellStatus(rowEl.find('.move-white'), row.white);
            if (row.white && row.white.valid) {
                gameState.fens.push(row.white.fen);
                lastValidFen = row.white.fen;
            } else if (row.white) {
                gameState.fens.push(row.white.fen || lastValidFen);
            } else {
                gameState.fens.push(lastValidFen);
            }

            // Black
            updateCellStatus(rowEl.find('.move-black'), row.black);
            if (row.black && row.black.valid) {
                gameState.fens.push(row.black.fen);
                lastValidFen = row.black.fen;
            } else if (row.black) {
                gameState.fens.push(row.black.fen || lastValidFen);
            } else {
                gameState.fens.push(lastValidFen);
            }
        });

        // Toggle Export Button
        $('#exportBtn').prop('disabled', !gameState.isValid);

        updateBoardStatus();

    } catch (err) {
        alert("Error uploading PGN: " + err.message);
    } finally {
        $('#loadingModal').addClass('hidden');
        // Clear input so same file can be uploaded again if needed
        $('#pgnInput').val('');
    }
}

// -- Drag & Drop Logic --

function onDragStart(source, piece, position, orientation) {
    if (game.game_over()) return false;
    if ((game.turn() === 'w' && piece.search(/^b/) !== -1) ||
        (game.turn() === 'b' && piece.search(/^w/) !== -1)) {
        return false;
    }
}

function onDrop(source, target) {
    const move = game.move({
        from: source,
        to: target,
        promotion: 'q'
    });

    if (move === null) return 'snapback';

    updateUIWithMove(move.san);
}

function onSnapEnd() {
    board.position(game.fen());
}

function updateUIWithMove(san) {
    const nextHalfMove = gameState.currentMoveIndex + 1;
    const rowIdx = Math.floor(nextHalfMove / 2);
    const isWhite = (nextHalfMove % 2) === 0;

    ensureRowExists(rowIdx);

    const row = $(`.move-row[data-idx="${rowIdx}"]`);
    const cell = isWhite ? row.find('.move-white input') : row.find('.move-black input');

    cell.val(san);
    validateMoves();
}

function ensureRowExists(rowIdx) {
    let row = $(`.move-row[data-idx="${rowIdx}"]`);
    if (row.length === 0) {
        const html = `
        <div class="grid grid-cols-[3rem_1fr_1fr] border-b border-gray-700 move-row" data-idx="${rowIdx}">
            <div class="py-2 text-center text-gray-500 font-mono text-sm move-num">${rowIdx + 1}.</div>
            
            <div class="move-cell move-white p-1">
                <input type="text" value="" 
                    class="move-input w-full h-full bg-transparent text-center focus:outline-none text-gray-200"
                    onchange="validateMoves()"
                    onfocus="highlightMove(${rowIdx}, 'white')">
            </div>
            
            <div class="move-cell move-black p-1 border-l border-gray-700">
                <input type="text" value="" 
                    class="move-input w-full h-full bg-transparent text-center focus:outline-none text-gray-200"
                    onchange="validateMoves()"
                    onfocus="highlightMove(${rowIdx}, 'black')">
            </div>
        </div>
        `;
        $('#movesGrid').append(html);
        const container = document.getElementById('movesGrid');
        container.scrollTop = container.scrollHeight;
    }
}
