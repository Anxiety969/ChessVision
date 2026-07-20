import time
import tkinter as tk
from pathlib import Path
from PIL import Image, ImageTk

import cv2
import numpy as np

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
def analyze_board():
    global last_analyzed_position
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
    display_board = cv2.resize(board, (320, 320))

    board_image = Image.fromarray(display_board)
    board_photo = ImageTk.PhotoImage(board_image)

    board_canvas.delete("all")
    board_canvas.create_image(0, 0, anchor="nw", image=board_photo)
    board_canvas.image = board_photo
    def mark_piece(square_name, outline):
        file_index = ord(square_name[0]) - ord("a")
        rank_index = 8 - int(square_name[1])

        x1 = file_index * 40 + 2
        y1 = rank_index * 40 + 2
        x2 = x1 + 36
        y2 = y1 + 36

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
            40,
        )

        options = selected_piece_options(
            recognized_position,
            square,
        )
        board_canvas.delete("move_highlight")
        def highlight_square(chess_square, outline, width):
            file_index = ord(chess_square[0]) - ord("a")
            rank_index = 8 - int(chess_square[1])

            x1 = file_index * 40
            y1 = rank_index * 40
            x2 = x1 + 40
            y2 = y1 + 40

            board_canvas.create_oval(
                x1 + 17,
                y1 + 17,
                x2 - 17,
                y2 - 17,
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

            x1 = file_index * 40
            y1 = rank_index * 40
            x2 = x1 + 40
            y2 = y1 + 40

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
    if auto_analyze_enabled.get():
        auto_analyze_tick()       
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


window = tk.Tk()
window.title("ChessVision")
window.geometry("500x980")
window.minsize(500, 980)
window.attributes("-topmost", True)

title_label = tk.Label(
    window,
    text="ChessVision",
    font=("Arial", 22, "bold"),
)
title_label.pack(pady=(10, 4))

instruction_label = tk.Label(
    window,
    text="Leave the board in the starting position.",
    font=("Arial", 11),
)
instruction_label.pack(pady=4)

capture_button = tk.Button(
    window,
    text="Create Piece Templates",
    font=("Arial", 12),
    command=capture_templates,
    padx=18,
    pady=10,
)
capture_button.pack(pady=4)
analyze_button = tk.Button(
    window,
    text="Analyze Board",
    font=("Arial", 12),
    command=analyze_board,
    padx=18,
    pady=10,
)

status_frame = tk.Frame(
    window,
    height=48,
)

status_frame.pack_propagate(False)

status_label = tk.Label(
    status_frame,
    text="Ready",
    font=("Arial", 10),
    justify="center",
    anchor="center",
    wraplength=460,
)
warning_frame = tk.LabelFrame(
    window,
    text="Training Warnings",
    font=("Arial", 11, "bold"),
    padx=6,
    pady=6,
)

warning_frame.pack(fill="x", padx=12, pady=8)
analyze_button.pack(
    before=warning_frame,
    pady=8,
)

warning_canvas = tk.Canvas(
    warning_frame,
    height=180,
    highlightthickness=0,
)

warning_scrollbar = tk.Scrollbar(
    warning_frame,
    orient="vertical",
    command=warning_canvas.yview,
)

warning_list_frame = tk.Frame(warning_canvas)

warning_list_window = warning_canvas.create_window(
    (0, 0),
    window=warning_list_frame,
    anchor="nw",
)

warning_canvas.configure(
    yscrollcommand=warning_scrollbar.set,
)

warning_canvas.pack(
    side="left",
    fill="both",
    expand=True,
)

warning_scrollbar.pack(
    side="right",
    fill="y",
)


def resize_warning_list(event):
    warning_canvas.itemconfigure(
        warning_list_window,
        width=event.width,
    )


def update_warning_scroll_region(event=None):
    warning_canvas.configure(
        scrollregion=warning_canvas.bbox("all"),
    )


warning_canvas.bind(
    "<Configure>",
    resize_warning_list,
)

warning_list_frame.bind(
    "<Configure>",
    update_warning_scroll_region,
)
danger_label = tk.Label(
    warning_list_frame,
    text="No immediate danger detected.",
    font=("Arial", 12, "bold"),
    fg="darkred",
    justify="left",
    anchor="w",
    wraplength=300,
)

danger_label.pack(
    fill="x",
    anchor="w",
)

opportunity_label = tk.Label(
    warning_list_frame,
    text="No immediate opportunity detected.",
    font=("Arial", 12, "bold"),
    fg="darkgreen",
    justify="left",
    anchor="w",
    wraplength=300,
)

opportunity_label.pack(
    fill="x",
    anchor="w",
    pady=(6, 0),
)
show_protected_squares = tk.BooleanVar(value=False)
auto_analyze_enabled = tk.BooleanVar(value=False)
protected_color = tk.StringVar(value="white")
protected_toggle = tk.Checkbutton(
    window,
    text="Show Protected Squares",
    variable=show_protected_squares,
)
auto_analyze_toggle = tk.Checkbutton(
    window,
    text="Auto Analyze",
    variable=auto_analyze_enabled,
    command=toggle_auto_analyze,
)

auto_analyze_toggle.pack(pady=4)
protected_color_frame = tk.Frame(window)

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
protected_toggle.pack(pady=4)
protected_color_frame.pack(pady=2)
status_frame.pack(fill="x", padx=12, pady=3)
status_label.pack(fill="both", expand=True)
board_canvas = tk.Canvas(
    window,
    width=320,
    height=320,
    bg="black",
    highlightthickness=1,
    highlightbackground="gray",
)
board_canvas.pack(pady=4)

window.mainloop()