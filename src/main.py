import time
import tkinter as tk
from pathlib import Path
from PIL import Image, ImageTk

import cv2
import numpy as np
import chess
import chess.engine

from vision import (
    capture_screen,
    compare_images,
    create_board_mask,
    identify_piece,
    load_template,
    square_to_coordinates,
    coordinates_to_square,
    knight_attacks,
    king_attacks,
    pawn_attacks,
    rook_attacks,
    bishop_attacks,
    queen_attacks,
    piece_attacks,
    color_attacks,
    king_in_check,
    checking_pieces,
    available_captures,
    attacked_pieces,
    hanging_pieces,
    knight_fork_opportunities,
    under_defended_pieces,
    make_move,
    move_is_legal,
    pinned_pieces,
    legal_moves_for_piece,
    legal_move_details,
    legal_captures_for_piece,
    all_legal_moves,
    selected_piece_options,
    pixel_to_square,

)   


PROJECT_FOLDER = Path(__file__).parent


STARTING_PIECES = {
    "a8": "black_rook",
    "b8": "black_knight",
    "c8": "black_bishop",
    "d8": "black_queen",
    "e8": "black_king",
    "f8": "black_bishop",
    "g8": "black_knight",
    "h8": "black_rook",

    "a7": "black_pawn",
    "b7": "black_pawn",
    "c7": "black_pawn",
    "d7": "black_pawn",
    "e7": "black_pawn",
    "f7": "black_pawn",
    "g7": "black_pawn",
    "h7": "black_pawn",

    "a2": "white_pawn",
    "b2": "white_pawn",
    "c2": "white_pawn",
    "d2": "white_pawn",
    "e2": "white_pawn",
    "f2": "white_pawn",
    "g2": "white_pawn",
    "h2": "white_pawn",

    "a1": "white_rook",
    "b1": "white_knight",
    "c1": "white_bishop",
    "d1": "white_queen",
    "e1": "white_king",
    "f1": "white_bishop",
    "g1": "white_knight",
    "h1": "white_rook",
}


def capture_templates():
    status_label.config(text="Capturing templates...")
    window.update_idletasks()

    window.withdraw()
    window.update()
    time.sleep(0.4)

    rgb_image = capture_screen()

    window.deiconify()
    window.update()

    board_mask = create_board_mask(rgb_image)

    kernel = np.ones((5, 5), dtype=np.uint8)

    cleaned_mask = cv2.morphologyEx(
        board_mask,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=2,
    )

    contours, _ = cv2.findContours(
        cleaned_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    candidates = []

    for contour in contours:
        x, y, width, height = cv2.boundingRect(contour)

        if width < 300 or height < 300:
            continue

        aspect_ratio = width / height

        if 0.85 <= aspect_ratio <= 1.15:
            candidates.append(
                (width * height, x, y, width, height)
            )

    if not candidates:
        status_label.config(text="No board found.")
        return

    candidates.sort(reverse=True)

    _, board_x, board_y, board_width, board_height = candidates[0]

    board = rgb_image[
        board_y:board_y + board_height,
        board_x:board_x + board_width,
    ]
    
    board_bgr = cv2.cvtColor(
    board,
    cv2.COLOR_RGB2BGR,
)
    square_width = board_width / 8
    square_height = board_height / 8

    files = "abcdefgh"
    ranks = "87654321"

    templates_folder = PROJECT_FOLDER / "piece_templates"
    templates_folder.mkdir(exist_ok=True)

    saved_names = set()

    for row in range(8):
        for column in range(8):
            square_name = f"{files[column]}{ranks[row]}"

            if square_name not in STARTING_PIECES:
                continue

            piece_name = STARTING_PIECES[square_name]
            if piece_name.endswith(("_pawn", "_rook", "_knight", "_bishop")):
                square_color = (
                    "light"
                    if (row + column) % 2 == 0
                    else "dark"
                )

                piece_name = f"{piece_name}_{square_color}"

            if piece_name in saved_names:
                continue

            x1 = round(column * square_width)
            y1 = round(row * square_height)
            x2 = round((column + 1) * square_width)
            y2 = round((row + 1) * square_height)

            square_image = board[y1:y2, x1:x2]

            square_bgr = cv2.cvtColor(
                square_image,
                cv2.COLOR_RGB2BGR,
            )

            output_path = (
                templates_folder /
                f"{piece_name}.png"
            )

            piece_templates[piece_name] = square_bgr

            cv2.imwrite(
                str(output_path),
                square_bgr,
            )

            saved_names.add(piece_name)

    status_label.config(
        text=(
            f"Saved {len(saved_names)} templates.\n"
            "Open the piece_templates folder."
        )
    )
last_analyzed_position = None
current_mode = "live"


PIECE_VALUES = {"p": 1, "n": 3, "b": 3, "r": 5, "q": 9, "k": 100}
PIECE_LETTERS = {"p": "", "n": "N", "b": "B", "r": "R", "q": "Q", "k": "K"}


def move_name(position, from_square, to_square):
    piece = position[from_square]
    capture = to_square in position
    prefix = PIECE_LETTERS[piece.lower()]
    if piece.lower() == "p" and capture:
        prefix = from_square[0]
    return f"{prefix}{'x' if capture else ''}{to_square}"


def quick_move_tiers(position, white):
    """Return fast, human-readable candidate moves without deep engine search."""
    candidates = []
    center = {"d4", "e4", "d5", "e5"}
    development_targets = {"c3", "f3", "c6", "f6", "c4", "f4", "c5", "f5"}

    for from_square, details in all_legal_moves(position, white).items():
        piece = position[from_square]
        for to_square in details["moves"] + [sq for sq, _ in details["captures"]]:
            captured = position.get(to_square)
            next_position = make_move(position, from_square, to_square)
            enemy_attacks = color_attacks(next_position, not white)
            safe = to_square not in enemy_attacks
            gives_check = king_in_check(next_position, not white)

            score = 0.0
            reasons = []
            if captured:
                gain = PIECE_VALUES[captured.lower()]
                cost = PIECE_VALUES[piece.lower()]
                score += gain * 3
                reasons.append(f"wins or trades for the {piece_names_for_tiers[captured.lower()]}")
                if safe:
                    score += 2
                elif gain < cost:
                    score -= (cost - gain) * 3
            if gives_check:
                score += 4
                reasons.append("gives check")
            if to_square in center:
                score += 1.5
                reasons.append("improves central control")
            if piece.lower() in {"n", "b"} and from_square[1] in {"1", "8"} and to_square in development_targets:
                score += 1.25
                reasons.append("develops a piece")
            if safe:
                score += 1
            else:
                score -= PIECE_VALUES[piece.lower()] * 0.8

            candidates.append({
                "from": from_square,
                "to": to_square,
                "name": move_name(position, from_square, to_square),
                "score": score,
                "safe": safe,
                "forcing": bool(captured or gives_check),
                "reason": reasons[0] if reasons else ("keeps the piece safe" if safe else "creates activity"),
            })

    candidates.sort(key=lambda item: item["score"], reverse=True)
    if not candidates:
        return []

    chosen = []
    best = candidates[0]
    chosen.append(("Best move", best))

    safer = next((m for m in candidates if m["safe"] and m["name"] != best["name"]), None)
    if safer:
        chosen.append(("Safer alternative", safer))

    aggressive = next((m for m in candidates if m["forcing"] and m["name"] not in {x[1]["name"] for x in chosen}), None)
    if aggressive:
        chosen.append(("Aggressive alternative", aggressive))

    return chosen[:3]


piece_names_for_tiers = {
    "p": "pawn", "n": "knight", "b": "bishop",
    "r": "rook", "q": "queen", "k": "king",
}
STOCKFISH_PATH = r"C:\Users\joshu\AppData\Local\Microsoft\WinGet\Packages\Stockfish.Stockfish_Microsoft.Winget.Source_8wekyb3d8bbwe\stockfish\stockfish-windows-x86-64-avx2.exe"

def stockfish_top_moves(position, white, count=3):
    """Return Stockfish's top legal moves for the recognized position."""
    board = chess.Board(None)

    for square_name, symbol in position.items():
        try:
            square = chess.parse_square(square_name)
            piece = chess.Piece.from_symbol(symbol)
        except (ValueError, TypeError):
            continue
        board.set_piece_at(square, piece)

    board.turn = chess.WHITE if white else chess.BLACK
    board.castling_rights = chess.BB_EMPTY
    board.ep_square = None
    board.halfmove_clock = 0
    board.fullmove_number = 1

    if board.king(chess.WHITE) is None or board.king(chess.BLACK) is None:
        return [], "Engine unavailable: both kings must be recognized."

    legal_count = board.legal_moves.count()
    if legal_count == 0:
        if board.is_checkmate():
            return [], "Checkmate — there are no legal moves."
        return [], "Stalemate — there are no legal moves."

    try:
        with chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH) as engine:
            results = engine.analyse(
                board,
                chess.engine.Limit(depth=14),
                multipv=min(count, legal_count),
            )
    except (OSError, chess.engine.EngineError, chess.engine.EngineTerminatedError) as error:
        return [], f"Stockfish error: {error}"

    if isinstance(results, dict):
        results = [results]

    moves = []
    for result in results:
        principal_variation = result.get("pv", [])
        if not principal_variation:
            continue

        move = principal_variation[0]
        san = board.san(move)
        score = result["score"].pov(board.turn)

        mate_in = score.mate()
        if mate_in is not None:
            evaluation = f"mate in {abs(mate_in)}" if mate_in > 0 else f"mated in {abs(mate_in)}"
        else:
            centipawns = score.score()
            evaluation = f"{centipawns / 100:+.2f}" if centipawns is not None else "unclear"

        moves.append({
            "san": san,
            "uci": move.uci(),
            "evaluation": evaluation,
        })

    return moves, None

def analyze_board():
    global last_analyzed_position, current_mode

    returning_from_practice = current_mode == "practice"
    current_mode = "live"

    if returning_from_practice:
        # Force the live position to be evaluated even when the physical board
        # has not changed since Practice Mode was opened.
        last_analyzed_position = None
        danger_label.config(text="Checking live position...", fg="gray40")
        opportunity_label.config(text="Checking live position...", fg="gray40")

    status_label.config(text="Analyzing board...")
    window.update_idletasks()

    if not auto_analyze_enabled.get():
        window.withdraw()
    window.update()
    time.sleep(0.4)

    rgb_image = capture_screen()

    if not auto_analyze_enabled.get():
        window.deiconify()
    window.update()
    window.update()

    board_mask = create_board_mask(rgb_image)
    kernel = np.ones((5, 5), dtype=np.uint8)

    cleaned_mask = cv2.morphologyEx(
    board_mask,
    cv2.MORPH_CLOSE,
    kernel,
    iterations=2,
)

    print("Board mask cleaned.")
    contours, _ = cv2.findContours(
    cleaned_mask,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE,
)

    print(f"Found {len(contours)} contours.")

    candidates = []
    for contour in contours:
        x, y, width, height = cv2.boundingRect(contour)

        if width < 300 or height < 300:
            continue

        aspect_ratio = width / height

        if 0.85 <= aspect_ratio <= 1.15:
            candidates.append(
                (width * height, x, y, width, height)
            )

    print(f"Found {len(candidates)} board candidates.")
    if not candidates:
        status_label.config(text="No board found.")
        return

    candidates.sort(reverse=True)

    _, board_x, board_y, board_width, board_height = candidates[0]

    print(
        f"Board found at {board_x}, {board_y} "
        f"with size {board_width}x{board_height}."
    )

    board = rgb_image[
    board_y:board_y + board_height,
    board_x:board_x + board_width,
]
    display_board = cv2.resize(board, (BOARD_DISPLAY_SIZE, BOARD_DISPLAY_SIZE))

    board_image = Image.fromarray(display_board)
    board_photo = ImageTk.PhotoImage(board_image)

    board_canvas.delete("all")
    board_canvas.create_image(0, 0, anchor="nw", image=board_photo)
    board_canvas.image = board_photo
    def mark_piece(square_name, outline):
        file_index = ord(square_name[0]) - ord("a")
        rank_index = 8 - int(square_name[1])

        x1 = file_index * SQUARE_DISPLAY_SIZE + 3
        y1 = rank_index * SQUARE_DISPLAY_SIZE + 3
        x2 = x1 + SQUARE_DISPLAY_SIZE - 6
        y2 = y1 + SQUARE_DISPLAY_SIZE - 6

        board_canvas.create_oval(
            x1,
            y1,
            x2,
            y2,
            outline=outline,
            width=4,
            tags="warning_highlight",
        )
    def handle_board_click(event):
        square = pixel_to_square(
            event.x,
            event.y,
            0,
            0,
            SQUARE_DISPLAY_SIZE,
        )

        options = selected_piece_options(
            recognized_position,
            square,
        )
        board_canvas.delete("move_highlight")
        def highlight_square(chess_square, outline, width):
            file_index = ord(chess_square[0]) - ord("a")
            rank_index = 8 - int(chess_square[1])

            x1 = file_index * SQUARE_DISPLAY_SIZE
            y1 = rank_index * SQUARE_DISPLAY_SIZE
            x2 = x1 + SQUARE_DISPLAY_SIZE
            y2 = y1 + SQUARE_DISPLAY_SIZE

            board_canvas.create_oval(
                x1 + SQUARE_DISPLAY_SIZE * 0.40,
                y1 + SQUARE_DISPLAY_SIZE * 0.40,
                x2 - SQUARE_DISPLAY_SIZE * 0.40,
                y2 - SQUARE_DISPLAY_SIZE * 0.40,
                fill=outline,
                outline="",
                tags="move_highlight",
            )

        highlight_square(square, "yellow", 4)

        for move in options["moves"]:
            highlight_square(move, "blue", 3)

        for capture_square, _ in options["captures"]:
            highlight_square(capture_square, "red", 3)

        print("Selected square:", square)
        print("Available moves:", options["moves"])
        print("Captures:", options["captures"])

    board_canvas.bind("<Button-1>", handle_board_click)
    def update_protected_squares():
        board_canvas.delete("protected_highlight")

        if not show_protected_squares.get():
            return

        protected_squares = color_attacks(
            recognized_position,
            protected_color.get() == "white",
        )
        for chess_square in protected_squares:
            if chess_square in recognized_position:
                continue
            file_index = ord(chess_square[0]) - ord("a")
            rank_index = 8 - int(chess_square[1])

            x1 = file_index * SQUARE_DISPLAY_SIZE
            y1 = rank_index * SQUARE_DISPLAY_SIZE
            x2 = x1 + SQUARE_DISPLAY_SIZE
            y2 = y1 + SQUARE_DISPLAY_SIZE

            board_canvas.create_rectangle(
                x1,
                y1,
                x2,
                y2,
                fill="#4da6ff",
                stipple="gray50",
                outline="#2f6fa8",
                width=1,
                tags="protected_highlight",
            )

    protected_toggle.config(command=update_protected_squares)
    print("Board cropped for analysis.")

    square_width = board_width / 8
    square_height = board_height / 8

    print(
        f"Each square is approximately "
        f"{square_width:.1f}x{square_height:.1f} pixels."
    )
    files = "abcdefgh"
    ranks = "87654321"

    print("Ready to scan all 64 squares.")
    recognized_position = {}
    piece_symbols = {
    "white_king": "K",
    "white_queen": "Q",
    "white_rook": "R",
    "white_bishop": "B",
    "white_knight": "N",
    "white_pawn": "P",
    "black_king": "k",
    "black_queen": "q",
    "black_rook": "r",
    "black_bishop": "b",
    "black_knight": "n",
    "black_pawn": "p",
}
    for row in range(8):
        for column in range(8):
            square_name = f"{files[column]}{ranks[row]}"
            x1 = round(column * square_width)
            y1 = round(row * square_height)
            x2 = round((column + 1) * square_width)
            y2 = round((row + 1) * square_height)

            square_image = board[y1:y2, x1:x2]

            square_bgr = cv2.cvtColor(
                square_image,
                cv2.COLOR_RGB2BGR,
            )

            piece_name, score = identify_piece(
                square_bgr,
                piece_templates,
            )
            if piece_name is not None:
                piece_name = piece_name.replace("_light", "")
                piece_name = piece_name.replace("_dark", "")
            if score < 0.50:
                print(square_name, "empty", f"{score:.3f}")
            else:
                recognized_position[square_name] = piece_symbols[piece_name]

                print(square_name, piece_name, f"{score:.3f}")

    current_position = tuple(sorted(recognized_position.items()))

    if (
        auto_analyze_enabled.get()
        and current_position == last_analyzed_position
    ):
        status_label.config(text="Watching for board changes...")
        return

    last_analyzed_position = current_position
    for rank in ranks:
        row_symbols = []

        for file in files:
            square_name = f"{file}{rank}"
            symbol = recognized_position.get(square_name, ".")
            row_symbols.append(symbol)

        print(" ".join(row_symbols))
    for square_name, symbol in recognized_position.items():
        print(
            f"{symbol} on {square_name} attacks:",
            piece_attacks(
                square_name,
                symbol,
                recognized_position,
            ),
        )
    print(
    "White attacks:",
    sorted(color_attacks(recognized_position, True)),
    )

    print(
        "Black attacks:",
        sorted(color_attacks(recognized_position, False)),
    )
    print(
    "White king in check:",
    king_in_check(recognized_position, True),
    )

    print(
        "Black king in check:",
        king_in_check(recognized_position, False),
    )    
    print(
    "Checking white king:",
    checking_pieces(recognized_position, True),
    )

    print(
    "Checking black king:",
    checking_pieces(recognized_position, False),
    )
    print(
    "White captures:",
    available_captures(recognized_position, True),
    )

    print(
        "Black captures:",
        available_captures(recognized_position, False),
    )
    print(
    "White pieces under attack:",
    attacked_pieces(recognized_position, True),
    )

    print(
        "Black pieces under attack:",
        attacked_pieces(recognized_position, False),
    )
    white_hanging = hanging_pieces(recognized_position, True)
    black_hanging = hanging_pieces(recognized_position, False)

    print("White hanging pieces:", white_hanging)
    print("Black hanging pieces:", black_hanging)

    piece_names = {
    "p": "pawn",
    "n": "knight",
    "b": "bishop",
    "r": "rook",
    "q": "queen",
    "k": "king",
}

    playing_white = protected_color.get() == "white"
    your_knight_forks = knight_fork_opportunities(
    recognized_position,
    playing_white,
)

    print("Your knight fork opportunities:", your_knight_forks)
    your_hanging = white_hanging if playing_white else black_hanging
    enemy_hanging = black_hanging if playing_white else white_hanging

    danger_lines = []
    opportunity_lines = []
    for start_square, target_square, attacked_targets in your_knight_forks:
        target_types = {
            symbol.lower()
            for _, symbol in attacked_targets
        }

        if not target_types.intersection({"k", "q", "r"}):
            continue

        attacked_names = [
            piece_names[symbol.lower()]
            for _, symbol in attacked_targets
        ]

        opportunity_lines.append(
            f"FORK AVAILABLE: Knight {start_square} to "
            f"{target_square} attacks the "
            f"{' and '.join(attacked_names)}."
        )

    for square_name, symbol in your_hanging:
        danger_lines.append(
            f"DANGER: Your {piece_names[symbol.lower()]} on "
            f"{square_name} is hanging!"
        )

    for square_name, symbol in enemy_hanging:
        opportunity_lines.append(
            f"OPPORTUNITY: Enemy {piece_names[symbol.lower()]} on "
            f"{square_name} is hanging."
        )

    if danger_lines:
        danger_label.config(
            text="\n".join(danger_lines),
            fg="darkred",
        )
    else:
        danger_label.config(
            text="No immediate danger detected.",
            fg="darkgreen",
        )

    if opportunity_lines:
        opportunity_label.config(
            text="\n".join(opportunity_lines),
            fg="darkgreen",
        )
    else:
        opportunity_label.config(
            text="No immediate opportunity detected.",
            fg="gray40",
        )
    print(
    "Test move e2 to d2 legal:",
    move_is_legal(
    recognized_position,
    "e2",
    "d2",
    ),
)
    print(
    "White pinned pieces:",
    pinned_pieces(recognized_position, True),
)

    print(
    "Black pinned pieces:",
    pinned_pieces(recognized_position, False),
)   
    print(
    "Legal moves for e2:",
    legal_moves_for_piece(
    recognized_position,
    "e2",
    ),
) 
    print(
    "Legal move details for e2:",
    legal_move_details(
        recognized_position,
        "e2",
    ),
)
    print(
    "Legal captures for e2:",
    legal_captures_for_piece(
        recognized_position,
        "e2",
    ),
)
    print(
    "All white legal moves:",
    all_legal_moves(
        recognized_position,
        True,
    ),
)

    print(
    "All black legal moves:",
    all_legal_moves(
        recognized_position,
        False,
    ),
)
    print(
    "Selected piece options e2:",
    selected_piece_options(
        recognized_position,
        "e2",
    ),
)
    print(
    "Pixel test:",
    pixel_to_square(
        900,
        300,
        826,
        281,
        213,
    ),
)
    white_under_defended = under_defended_pieces(
        recognized_position,
        True,
    )

    black_under_defended = under_defended_pieces(
        recognized_position,
        False,
    )

    print(
        "White under-defended pieces:",
        white_under_defended,
    )

    print(
        "Black under-defended pieces:",
        black_under_defended,
    )

    your_under_defended = (
        white_under_defended
        if playing_white
        else black_under_defended
    )

    enemy_under_defended = (
        black_under_defended
        if playing_white
        else white_under_defended
    )

    for square_name, symbol, attackers in your_under_defended:
                if (square_name, symbol) in your_hanging:
                    continue
                attacker_names = ", ".join(
            f"{piece_names[attacker_symbol.lower()]} on {attacker_square}"
            for attacker_square, attacker_symbol in attackers
        )

                danger_lines.append(
            f"WARNING: Your {piece_names[symbol.lower()]} on "
            f"{square_name} is under-defended.\n"
            f"Attacked by: {attacker_names}"
        )

    for square_name, symbol, attackers in enemy_under_defended:
                if (square_name, symbol) in enemy_hanging:
                    continue
                attacker_names = ", ".join(
            f"{piece_names[attacker_symbol.lower()]} on {attacker_square}"
            for attacker_square, attacker_symbol in attackers
        )

                opportunity_lines.append(
            f"TARGET: Enemy {piece_names[symbol.lower()]} on "
            f"{square_name} is under-defended.\n"
            f"Attacked by: {attacker_names}"
        )

    # Ask Stockfish for the three strongest legal moves in the position.
    tier_lines = []
    engine_moves, engine_error = stockfish_top_moves(
        recognized_position,
        playing_white,
        count=3,
    )

    if engine_moves:
        tier_lines.append("STOCKFISH — TOP 3 MOVES")
        for index, move in enumerate(engine_moves, start=1):
            tier_lines.append(
                f"{index}. {move['san']}  ({move['evaluation']})"
            )
    elif engine_error:
        tier_lines.append(engine_error)

    sound_forks = []
    for fork in your_knight_forks:
        start_square, target_square, attacked_targets = fork
        fork_position = make_move(recognized_position, start_square, target_square)
        destination_safe = target_square not in color_attacks(fork_position, not playing_white)
        target_value = sum(PIECE_VALUES[symbol.lower()] for _, symbol in attacked_targets)
        if destination_safe or target_value > PIECE_VALUES["n"]:
            sound_forks.append(fork)

    if sound_forks:
        first_fork = sound_forks[0]
        fork_move = move_name(recognized_position, first_fork[0], first_fork[1])
        tier_lines.append(f"Tactical line: {fork_move} creates a fork worth examining.")

    if your_hanging:
        square_name, symbol = your_hanging[0]
        tier_lines.append(
            f"If you ignore the threat: your {piece_names[symbol.lower()]} on "
            f"{square_name} can be captured."
        )
    elif your_under_defended:
        square_name, symbol, _ = your_under_defended[0]
        tier_lines.append(
            f"If you ignore the threat: your {piece_names[symbol.lower()]} on "
            f"{square_name} may be lost in an unfavorable trade."
        )

    opportunity_lines = tier_lines + opportunity_lines

    danger_label.config(
        text="\n\n".join(danger_lines)
        if danger_lines
        else "No immediate danger detected.",
        fg="darkred" if danger_lines else "darkgreen",
    )

    opportunity_label.config(
        text="\n\n".join(opportunity_lines)
        if opportunity_lines
        else "No immediate opportunity detected.",
        fg="darkgreen" if opportunity_lines else "gray40",
    )
    board_canvas.delete("warning_highlight")

    for square_name, _ in your_hanging:
        mark_piece(square_name, "red")

    for square_name, _ in enemy_hanging:
        mark_piece(square_name, "green")

    for square_name, symbol, _ in your_under_defended:
        if (square_name, symbol) not in your_hanging:
            mark_piece(square_name, "orange")
def auto_analyze_tick():
    if not auto_analyze_enabled.get():
        return

    analyze_board()
    window.after(2000, auto_analyze_tick)


def toggle_auto_analyze():
    global current_mode, last_analyzed_position
    if auto_analyze_enabled.get():
        if current_mode == "practice":
            # Do not let the unchanged-position shortcut preserve lesson text.
            last_analyzed_position = None
            danger_label.config(text="Checking live position...", fg="gray40")
            opportunity_label.config(text="Checking live position...", fg="gray40")

        current_mode = "live"
        status_label.config(text="Watching for board changes...")
        auto_analyze_tick()

LONDON_POSITIONS = [
    {
        "name": "London Setup — Develop the bishop",
        "fen": "rnbqkbnr/ppp1pppp/8/3p4/3P1B2/8/PPP1PPPP/RN1QKBNR b KQkq - 1 2",
        "warning": "Black may challenge your center with ...c5 or ...e5.",
        "options": [
            "Best move: Nf3 — develop and support the center.",
            "Safer alternative: e3 — secure d4 and open the bishop.",
            "Aggressive alternative: Nc3 — adds pressure on d5.",
        ],
    },
    {
        "name": "London Setup — Build the triangle",
        "fen": "rnbqkb1r/ppp1pppp/5n2/3p4/3P1B2/4P3/PPP2PPP/RN1QKBNR w KQkq - 1 3",
        "warning": "Do not allow ...Nh5 to win the bishop without a plan.",
        "options": [
            "Best move: Nf3 — finish the standard setup.",
            "Safer alternative: h3 — gives the bishop an escape square.",
            "Aggressive alternative: c4 — immediately challenge d5.",
        ],
    },
    {
        "name": "London Setup — Choose the plan",
        "fen": "rnbqkb1r/ppp1pppp/5n2/3p4/3P1B2/2N1PN2/PPP2PPP/R2QKB1R b KQkq - 3 4",
        "warning": "Watch for ...c5 followed by ...Qb6 against b2.",
        "options": [
            "Best move: Bd3 — point the bishop toward h7.",
            "Safer alternative: Be2 — prepare quick castling.",
            "Aggressive alternative: Nb5 — pressure c7 when justified.",
        ],
    },
]

CHESS_SYMBOLS = {
    "K": "♔", "Q": "♕", "R": "♖", "B": "♗", "N": "♘", "P": "♙",
    "k": "♚", "q": "♛", "r": "♜", "b": "♝", "n": "♞", "p": "♟",
}

london_position_index = -1


def parse_fen_board(fen):
    position = {}
    ranks = fen.split()[0].split("/")
    for row, rank_data in enumerate(ranks):
        file_index = 0
        for character in rank_data:
            if character.isdigit():
                file_index += int(character)
                continue
            square = f"{chr(ord('a') + file_index)}{8 - row}"
            position[square] = character
            file_index += 1
    return position


def draw_training_board(fen):
    board_canvas.delete("all")
    light_square = "#f0d9b5"
    dark_square = "#b58863"
    position = parse_fen_board(fen)

    for row in range(8):
        for column in range(8):
            x1 = column * SQUARE_DISPLAY_SIZE
            y1 = row * SQUARE_DISPLAY_SIZE
            x2 = x1 + SQUARE_DISPLAY_SIZE
            y2 = y1 + SQUARE_DISPLAY_SIZE
            fill = light_square if (row + column) % 2 == 0 else dark_square
            board_canvas.create_rectangle(x1, y1, x2, y2, fill=fill, outline=fill)

    for square, piece in position.items():
        column = ord(square[0]) - ord("a")
        row = 8 - int(square[1])
        center_x = column * SQUARE_DISPLAY_SIZE + SQUARE_DISPLAY_SIZE / 2
        center_y = row * SQUARE_DISPLAY_SIZE + SQUARE_DISPLAY_SIZE / 2
        board_canvas.create_text(
            center_x,
            center_y,
            text=CHESS_SYMBOLS[piece],
            font=("Segoe UI Symbol", 40),
        )


def practice_london():
    global london_position_index, current_mode

    # Practice mode owns the board until live analysis is requested again.
    current_mode = "practice"
    auto_analyze_enabled.set(False)

    london_position_index = (london_position_index + 1) % len(LONDON_POSITIONS)
    lesson = LONDON_POSITIONS[london_position_index]

    draw_training_board(lesson["fen"])
    status_label.config(
        text=(
            f"London practice {london_position_index + 1}/{len(LONDON_POSITIONS)}: "
            f"{lesson['name']} — Analyze Board or Auto Analyze returns to live mode."
        )
    )
    danger_label.config(text=lesson["warning"], fg="darkred")
    opportunity_label.config(text="\n\n".join(lesson["options"]), fg="darkgreen")


piece_templates = {
"white_pawn_light": load_template("white_pawn_light"),
"white_pawn_dark": load_template("white_pawn_dark"),
"white_knight_light": load_template("white_knight_light"),
"white_knight_dark": load_template("white_knight_dark"),
"white_bishop": load_template("white_bishop"),
"white_rook_light": load_template("white_rook_light"),
"white_rook_dark": load_template("white_rook_dark"),
"white_queen": load_template("white_queen"),
"white_king": load_template("white_king"),
"black_pawn_light": load_template("black_pawn_light"),
"black_pawn_dark": load_template("black_pawn_dark"),
"black_knight_light": load_template("black_knight_light"),
"black_knight_dark": load_template("black_knight_dark"),
"black_bishop": load_template("black_bishop"),
"black_rook_light": load_template("black_rook_light"),
"black_rook_dark": load_template("black_rook_dark"),
"black_queen": load_template("black_queen"),
"black_king": load_template("black_king"),
}


BOARD_DISPLAY_SIZE = 440
SQUARE_DISPLAY_SIZE = BOARD_DISPLAY_SIZE // 8

window = tk.Tk()
window.title("ChessVision")
window.geometry("520x920")
window.minsize(520, 920)
window.attributes("-topmost", True)

# Compact heading so the board remains the visual focus.
header_frame = tk.Frame(window)
header_frame.pack(fill="x", padx=10, pady=(6, 2))

title_label = tk.Label(
    header_frame,
    text="ChessVision",
    font=("Arial", 16, "bold"),
)
title_label.pack(side="left")

instruction_label = tk.Label(
    header_frame,
    text="Live board coach",
    font=("Arial", 9),
    fg="gray40",
)
instruction_label.pack(side="right")

# The board is the primary element and appears first.
board_canvas = tk.Canvas(
    window,
    width=BOARD_DISPLAY_SIZE,
    height=BOARD_DISPLAY_SIZE,
    bg="black",
    highlightthickness=2,
    highlightbackground="gray35",
)
board_canvas.pack(pady=(2, 5))

status_frame = tk.Frame(window, height=38)
status_frame.pack_propagate(False)
status_frame.pack(fill="x", padx=12, pady=(0, 4))

status_label = tk.Label(
    status_frame,
    text="Ready",
    font=("Arial", 10),
    justify="center",
    anchor="center",
    wraplength=490,
)
status_label.pack(fill="both", expand=True)

show_protected_squares = tk.BooleanVar(value=False)
auto_analyze_enabled = tk.BooleanVar(value=False)
protected_color = tk.StringVar(value="white")

controls_frame = tk.Frame(window)
controls_frame.pack(fill="x", padx=12, pady=(0, 5))

analyze_button = tk.Button(
    controls_frame,
    text="Analyze Board",
    font=("Arial", 11, "bold"),
    command=analyze_board,
    padx=12,
    pady=5,
)
analyze_button.pack(side="left", padx=(0, 8))

london_button = tk.Button(
    controls_frame,
    text="Practice London",
    command=practice_london,
    padx=8,
    pady=5,
)
london_button.pack(side="left", padx=(0, 8))

auto_analyze_toggle = tk.Checkbutton(
    controls_frame,
    text="Auto Analyze",
    variable=auto_analyze_enabled,
    command=toggle_auto_analyze,
)
auto_analyze_toggle.pack(side="left", padx=(0, 8))

protected_toggle = tk.Checkbutton(
    controls_frame,
    text="Protected Squares",
    variable=show_protected_squares,
)
protected_toggle.pack(side="left")

protected_color_frame = tk.Frame(window)
protected_color_frame.pack(pady=(0, 5))

tk.Radiobutton(
    protected_color_frame,
    text="White",
    variable=protected_color,
    value="white",
    command=lambda: (
        protected_toggle.invoke(),
        protected_toggle.invoke(),
    ) if show_protected_squares.get() else None,
).pack(side="left")

tk.Radiobutton(
    protected_color_frame,
    text="Black",
    variable=protected_color,
    value="black",
    command=lambda: (
        protected_toggle.invoke(),
        protected_toggle.invoke(),
    ) if show_protected_squares.get() else None,
).pack(side="left")

# Separate fixed-height, scrollable panes for danger and opportunity text.
message_row = tk.Frame(window)
message_row.pack(fill="both", expand=True, padx=10, pady=(0, 8))
message_row.grid_columnconfigure(0, weight=1)
message_row.grid_columnconfigure(1, weight=1)
message_row.grid_rowconfigure(0, weight=1)

warning_frame = tk.LabelFrame(
    message_row,
    text="Warnings!",
    font=("Arial", 11, "bold"),
    padx=4,
    pady=4,
)
warning_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 4))

opportunity_frame = tk.LabelFrame(
    message_row,
    text="Opportunities!",
    font=("Arial", 11, "bold"),
    padx=4,
    pady=4,
)
opportunity_frame.grid(row=0, column=1, sticky="nsew", padx=(4, 0))


def build_scrollable_message_box(parent):
    canvas = tk.Canvas(parent, height=150, highlightthickness=0)
    scrollbar = tk.Scrollbar(parent, orient="vertical", command=canvas.yview)
    content = tk.Frame(canvas)
    content_window = canvas.create_window((0, 0), window=content, anchor="nw")
    canvas.configure(yscrollcommand=scrollbar.set)
    canvas.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")

    def resize_content(event):
        canvas.itemconfigure(content_window, width=event.width)

    def update_scroll_region(event=None):
        canvas.configure(scrollregion=canvas.bbox("all"))

    canvas.bind("<Configure>", resize_content)
    content.bind("<Configure>", update_scroll_region)
    return content


warning_list_frame = build_scrollable_message_box(warning_frame)
opportunity_list_frame = build_scrollable_message_box(opportunity_frame)

danger_label = tk.Label(
    warning_list_frame,
    text="No immediate danger detected.",
    font=("Arial", 10, "bold"),
    fg="darkgreen",
    justify="left",
    anchor="nw",
    wraplength=205,
)
danger_label.pack(fill="x", anchor="w")

opportunity_label = tk.Label(
    opportunity_list_frame,
    text="No immediate opportunity detected.",
    font=("Arial", 10, "bold"),
    fg="gray40",
    justify="left",
    anchor="nw",
    wraplength=205,
)
opportunity_label.pack(fill="x", anchor="w")

window.mainloop()