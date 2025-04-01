import datetime  # Import the datetime module to generate timestamps for PGN files
import os  # Import the os module for interacting with the operating system (e.g., file paths)

class Move:
    def __init__(self, start, end, piece, captured_piece=None, promotion=None, is_castling=False):
        self.start = start  # Set the starting position of the move as a tuple (row, col)
        self.end = end  # Set the ending position of the move as a tuple (row, col)
        self.piece = piece  # Set the piece being moved (e.g., "wP" for white pawn)
        self.captured_piece = captured_piece  # Set the captured piece, if any (e.g., "bN" for black knight, or None)
        self.promotion = promotion  # Set the promotion choice, if applicable (e.g., "Q" for queen, or None)
        self.is_castling = is_castling  # Set whether the move is a castling move (True or False)
        self.score = 0  # Initialize a score for move ordering during AI search (default 0)
        
    def __str__(self):
        return f"{self.piece} from {self.start} to {self.end}"  # Define a string representation of the move (e.g., "wP from (1, 4) to (3, 4)")

    def __eq__(self, other):
        if not isinstance(other, Move):  # Check if the other object is an instance of the Move class
            return False  # Return False if the other object is not a Move
        return (self.start == other.start and self.end == other.end and
                self.piece == other.piece and self.captured_piece == other.captured_piece and
                self.promotion == other.promotion and self.is_castling == other.is_castling)  # Compare all attributes of the move to determine equality

class ChessEngine:
    def __init__(self, fen=None):
        self.turn = "w"  # Set the initial turn to White ("w")
        self.moved_pieces = set()  # Initialize an empty set to track pieces that have moved (for castling)
        self.previous_move = None  # Initialize the previous move as None
        self.en_passant_target = None  # Initialize the en passant target square as None
        self.move_history = []  # Initialize an empty list to store the move history in algebraic notation
        self.move_number = 1  # Set the initial move number to 1
        self.state_history = []  # Initialize an empty list to store game states for undoing moves
        self.halfmove_clock = 0  # Initialize the halfmove clock for the 50-move rule (counts moves without captures or pawn moves)
        self.position_history = {}  # Initialize a dictionary to track position occurrences for threefold repetition
        self.promotion_pending = None  # Initialize the promotion pending flag as None (set when a pawn reaches the 8th rank)
        # Add move counter
        self.moves_evaluated = 0
        if fen:  # Check if a FEN string is provided
            self.board = self.load_fen(fen)  # Load the board from the provided FEN string
        else:
            self.board = self.load_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w - - 0 1")  # Load the default starting position using FEN

        # Piece values and tables from CEngine
        self.piece_values = {
            'P': 100, 'N': 320, 'B': 330, 'R': 500, 'Q': 900, 'K': 20000
        }  # Define a dictionary of piece values for evaluation (Pawn=100, Knight=320, etc.)
        self.pawn_table = [
            0,  0,  0,  0,  0,  0,  0,  0,
            50, 50, 50, 50, 50, 50, 50, 50,
            10, 10, 20, 30, 30, 20, 10, 10,
            5,  5, 10, 25, 25, 10,  5,  5,
            0,  0,  0, 20, 20,  0,  0,  0,
            5, -5,-10,  0,  0,-10, -5,  5,
            5, 10, 10,-20,-20, 10, 10,  5,
            0,  0,  0,  0,  0,  0,  0,  0
        ]  # Define a position table for pawns to evaluate their placement (higher values for better squares)
        self.knight_table = [
            -50,-40,-30,-30,-30,-30,-40,-50,
            -40,-20,  0,  0,  0,  0,-20,-40,
            -30,  0, 10, 15, 15, 10,  0,-30,
            -30,  5, 15, 20, 20, 15,  5,-30,
            -30,  0, 15, 20, 20, 15,  0,-30,
            -30,  5, 10, 15, 15, 10,  5,-30,
            -40,-20,  0,  5,  5,  0,-20,-40,
            -50,-40,-30,-30,-30,-30,-40,-50
        ]  # Define a position table for knights to evaluate their placement (higher values for better squares)

    def load_fen(self, fen):
        board = [[None for _ in range(8)] for _ in range(8)]
        parts = fen.split()
        position = parts[0]
        active_color = parts[1]
        castling = parts[2]
        en_passant = parts[3]
        self.halfmove_clock = int(parts[4])
        self.move_number = int(parts[5])
        self.turn = active_color
        self.searching = False

        rows = position.split("/")
        for row in range(8):
            col = 0
            # Map FEN rank 8 (rows[0]) to board row 0, rank 1 (rows[7]) to board row 7
            for char in rows[row]:  # Remove the 7 - row inversion
                if char.isdigit():
                    for _ in range(int(char)):
                        board[row][col] = "."
                        col += 1
                else:
                    piece_color = "w" if char.isupper() else "b"
                    piece_type = char.upper()
                    board[row][col] = piece_color + piece_type
                    col += 1

        if castling != "-":  # Check if castling is available (not "-")
            for char in castling:  # Loop through each character in the castling string
                if char == "K":  # Check if White can castle kingside
                    self.moved_pieces.discard((0, 4))  # Ensure the White king hasn't moved
                    self.moved_pieces.discard((0, 7))  # Ensure the White kingside rook hasn't moved
                elif char == "Q":  # Check if White can castle queenside
                    self.moved_pieces.discard((0, 4))  # Ensure the White king hasn't moved
                    self.moved_pieces.discard((0, 0))  # Ensure the White queenside rook hasn't moved
                elif char == "k":  # Check if Black can castle kingside
                    self.moved_pieces.discard((7, 4))  # Ensure the Black king hasn't moved
                    self.moved_pieces.discard((7, 7))  # Ensure the Black kingside rook hasn't moved
                elif char == "q":  # Check if Black can castle queenside
                    self.moved_pieces.discard((7, 4))  # Ensure the Black king hasn't moved
                    self.moved_pieces.discard((7, 0))  # Ensure the Black queenside rook hasn't moved

        if en_passant != "-":  # Check if there is an en passant target square
            files = "abcdefgh"  # Define the file letters (a to h)
            col = files.index(en_passant[0])  # Convert the file letter to a column index (0 to 7)
            row = 8 - int(en_passant[1])  # Convert the rank to a row index (FEN uses 1-8, board uses 0-7)
            self.en_passant_target = (row, col)  # Set the en passant target square
        else:
            self.en_passant_target = None  # Set the en passant target to None if not available

        return board  # Return the constructed board

    def get_fen(self):
        fen = ""  # Initialize an empty string to build the FEN
        for row in range(7, -1, -1):  # Loop through rows from 7 to 0 (FEN starts from rank 8)
            empty_count = 0  # Initialize a counter for consecutive empty squares
            for col in range(8):  # Loop through each column in the row
                piece = self.board[row][col]  # Get the piece at the current position
                if piece == ".":  # Check if the square is empty
                    empty_count += 1  # Increment the empty square counter
                else:  # If the square has a piece
                    if empty_count > 0:  # Check if there were any empty squares before this piece
                        fen += str(empty_count)  # Add the number of empty squares to the FEN
                        empty_count = 0  # Reset the empty square counter
                    if piece[0] == "w":  # Check if the piece is white
                        fen += piece[1].upper()  # Add the piece type in uppercase (e.g., "P" for pawn)
                    else:  # If the piece is black
                        fen += piece[1].lower()  # Add the piece type in lowercase (e.g., "p" for pawn)
            if empty_count > 0:  # Check if there are any remaining empty squares at the end of the row
                fen += str(empty_count)  # Add the number of empty squares to the FEN
            if row > 0:  # Check if this is not the last row
                fen += "/"  # Add a "/" to separate rows in the FEN

        active_color = "b" if self.turn == "w" else "w"  # Determine the active color (opposite of current turn for FEN)
        fen += " " + active_color  # Add the active color to the FEN

        castling = ""  # Initialize a string to store castling availability
        if (0, 4) not in self.moved_pieces:  # Check if the White king hasn't moved
            if (0, 7) not in self.moved_pieces:  # Check if the White kingside rook hasn't moved
                castling += "K"  # Add "K" to indicate White can castle kingside
            if (0, 0) not in self.moved_pieces:  # Check if the White queenside rook hasn't moved
                castling += "Q"  # Add "Q" to indicate White can castle queenside
        if (7, 4) not in self.moved_pieces:  # Check if the Black king hasn't moved
            if (7, 7) not in self.moved_pieces:  # Check if the Black kingside rook hasn't moved
                castling += "k"  # Add "k" to indicate Black can castle kingside
            if (7, 0) not in self.moved_pieces:  # Check if the Black queenside rook hasn't moved
                castling += "q"  # Add "q" to indicate Black can castle queenside
        fen += " " + (castling if castling else "-")  # Add the castling availability to the FEN (or "-" if none)

        if self.en_passant_target:  # Check if there is an en passant target square
            row, col = self.en_passant_target  # Get the row and column of the en passant target
            files = "abcdefgh"  # Define the file letters (a to h)
            fen += " " + files[col] + str(8 - row)  # Add the en passant target square in algebraic notation (e.g., "e3")
        else:
            fen += " -"  # Add "-" if there is no en passant target

        fen += f" {self.halfmove_clock} {self.move_number}"  # Add the halfmove clock and move number to the FEN
        return fen  # Return the complete FEN string

    def to_algebraic(self, move):
        start, end = move.start, move.end
        piece = move.piece
        x, y = start  # Start row, col
        ex, ey = end  # End row, col
        files = "abcdefgh"
        move_str = ""
        
        if piece[1] == "P":
            if y != ey:  # Capture (including en passant)
                move_str += files[y] + "x"
            move_str += f"{files[ey]}{8 - ex}"
            if move.promotion:
                move_str += "=" + move.promotion
        else:
            move_str += piece[1]  # Piece type (N, B, R, Q, K)

            if piece[1] in ("N", "R"):  # Disambiguation for Knights and Rooks
                same_type_moves = []
                # Find all other pieces of the same type and color that can move to the same square
                for row in range(8):
                    for col in range(8):
                        if (row, col) != start and self.board[row][col] == piece:
                            moves = self.get_piece_moves(row, col)
                            for m in moves:
                                if self.is_legal_move(m.start, m.end) and m.end == end:
                                    same_type_moves.append(m)

                if same_type_moves:  # If there’s at least one other piece that can move to the same square
                    ambiguous_file = False
                    ambiguous_rank = False
                    for m in same_type_moves:
                        if m.start[1] == y:  # Same file
                            ambiguous_file = True
                        if m.start[0] == x:  # Same rank
                            ambiguous_rank = True

                    # Determine disambiguation based on ambiguity
                    if ambiguous_file and ambiguous_rank:
                        # Both file and rank are the same for some other piece (rare, e.g., three knights)
                        move_str += files[y] + str(8 - x)
                    elif ambiguous_file:
                        # Same file as another piece, use rank
                        move_str += str(8 - x)
                    elif ambiguous_rank:
                        # Same rank as another piece, use file
                        move_str += files[y]
                    else:
                        # Different file and rank from all others, file is sufficient
                        move_str += files[y]

            if move.captured_piece:
                move_str += "x"
            move_str += f"{files[ey]}{8 - ex}"
            if move.is_castling:
                move_str = "O-O" if ey == 6 else "O-O-O"

        return move_str

    def get_position_key(self):
        fen = ""  # Initialize an empty string to build the position key (similar to FEN but for repetition tracking)
        for row in range(7, -1, -1):  # Loop through rows from 7 to 0
            empty_count = 0  # Initialize a counter for consecutive empty squares
            for col in range(8):  # Loop through each column in the row
                piece = self.board[row][col]  # Get the piece at the current position
                if piece == ".":  # Check if the square is empty
                    empty_count += 1  # Increment the empty square counter
                else:  # If the square has a piece
                    if empty_count > 0:  # Check if there were any empty squares before this piece
                        fen += str(empty_count)  # Add the number of empty squares to the key
                        empty_count = 0  # Reset the empty square counter
                    if piece[0] == "w":  # Check if the piece is white
                        fen += piece[1].upper()  # Add the piece type in uppercase
                    else:  # If the piece is black
                        fen += piece[1].lower()  # Add the piece type in lowercase
            if empty_count > 0:  # Check if there are any remaining empty squares at the end of the row
                fen += str(empty_count)  # Add the number of empty squares to the key
            if row > 0:  # Check if this is not the last row
                fen += "/"  # Add a "/" to separate rows

        active_color = "b" if self.turn == "w" else "w"  # Determine the active color (opposite of current turn)
        fen += " " + active_color  # Add the active color to the key

        castling = ""  # Initialize a string to store castling availability
        if (0, 4) not in self.moved_pieces:  # Check if the White king hasn't moved
            if (0, 7) not in self.moved_pieces:  # Check if the White kingside rook hasn't moved
                castling += "K"  # Add "K" to indicate White can castle kingside
            if (0, 0) not in self.moved_pieces:  # Check if the White queenside rook hasn't moved
                castling += "Q"  # Add "Q" to indicate White can castle queenside
        if (7, 4) not in self.moved_pieces:  # Check if the Black king hasn't moved
            if (7, 7) not in self.moved_pieces:  # Check if the Black kingside rook hasn't moved
                castling += "k"  # Add "k" to indicate Black can castle kingside
            if (7, 0) not in self.moved_pieces:  # Check if the Black queenside rook hasn't moved
                castling += "q"  # Add "q" to indicate Black can castle queenside
        fen += " " + (castling if castling else "-")  # Add the castling availability to the key (or "-" if none)

        if self.en_passant_target:  # Check if there is an en passant target square
            row, col = self.en_passant_target  # Get the row and column of the en passant target
            files = "abcdefgh"  # Define the file letters (a to h)
            fen += " " + files[col] + str(8 - row)  # Add the en passant target square in algebraic notation
        else:
            fen += " -"  # Add "-" if there is no en passant target

        return fen  # Return the position key (used for threefold repetition)

    def save_state(self):
        state = {
            "board": [row[:] for row in self.board],
            "turn": self.turn,
            "moved_pieces": self.moved_pieces.copy(),
            "previous_move": self.previous_move,
            "en_passant_target": self.en_passant_target,
            "move_history": self.move_history.copy(),
            "move_number": self.move_number,
            "halfmove_clock": self.halfmove_clock,
        }
        self.state_history.append(state)  # Append once, outside the loop     

    def evaluate_position(self):
        if self.is_game_over()[0]:
            result = self.is_game_over()[1]
            if "White wins" in result:
                return 100000
            elif "Black wins" in result:
                return -100000
            return 0

        material_score = 0
        position_score = 0
        center_control = 0

        # Center squares (d4, d5, e4, e5)
        center_squares = [(3, 3), (3, 4), (4, 3), (4, 4)]

        for row in range(8):
            for col in range(8):
                piece = self.board[row][col]
                if piece != ".":
                    color = piece[0]
                    piece_type = piece[1]
                    value = self.piece_values[piece_type]
                    square = row * 8 + col
                    
                    if color == 'w':
                        material_score += value
                        if piece_type == 'P':
                            position_score += self.pawn_table[square]
                            if (row, col) in center_squares:
                                center_control += 10  # Bonus for controlling the center
                        elif piece_type == 'N':
                            position_score += self.knight_table[square]
                            if (row, col) in center_squares:
                                center_control += 15
                    else:
                        material_score -= value
                        if piece_type == 'P':
                            position_score -= self.pawn_table[63 - square]
                            if (row, col) in center_squares:
                                center_control -= 10
                        elif piece_type == 'N':
                            position_score -= self.knight_table[63 - square]
                            if (row, col) in center_squares:
                                center_control -= 15

        king_safety = self.evaluate_king_safety()
        pawn_structure = self.evaluate_pawn_structure()
        return material_score + position_score + king_safety + pawn_structure + center_control

    def evaluate_king_safety(self):
        """Basic king safety evaluation."""
        score = 0  # Initialize the king safety score
        for color in ['w', 'b']:  # Loop through both colors (white and black)
            king_pos = None  # Initialize the king's position as None
            for row in range(8):  # Loop through each row of the board
                for col in range(8):  # Loop through each column of the board
                    if self.board[row][col] == f"{color}K":  # Check if the current square has the king of the given color
                        king_pos = (row, col)  # Set the king's position
                        break  # Exit the column loop
                if king_pos:  # Check if the king was found
                    break  # Exit the row loop
            
            if king_pos:  # Check if the king was found
                x, y = king_pos  # Get the king's row and column
                if 2 <= x <= 5 and 2 <= y <= 5:  # Check if the king is in the center of the board
                    score += -20 if color == 'w' else 20  # Penalize White's king or reward Black's king for being in the center
                
                direction = -1 if color == 'w' else 1  # Set the direction to check for pawns (up for White, down for Black)
                for dy in [-1, 0, 1]:  # Check the three squares in front of the king (left, center, right)
                    if 0 <= x + direction < 8 and 0 <= y + dy < 8:  # Check if the square is within the board
                        if self.board[x + direction][y + dy] == f"{color}P":  # Check if there is a pawn of the same color in front
                            score += 10 if color == 'w' else -10  # Reward White or penalize Black for having a pawn shield
        return score  # Return the king safety score

    def evaluate_pawn_structure(self):
        """Basic pawn structure evaluation."""
        score = 0  # Initialize the pawn structure score
        for col in range(8):  # Loop through each column of the board
            white_pawns = sum(1 for row in range(8) if self.board[row][col] == 'wP')  # Count White pawns in the column
            black_pawns = sum(1 for row in range(8) if self.board[row][col] == 'bP')  # Count Black pawns in the column
            
            if white_pawns > 1:  # Check if there are doubled White pawns
                score -= 20 * (white_pawns - 1)  # Penalize White for each extra pawn in the column
            if black_pawns > 1:  # Check if there are doubled Black pawns
                score += 20 * (black_pawns - 1)  # Penalize Black for each extra pawn in the column
            
            if white_pawns > 0 and col > 0 and col < 7:  # Check if there are White pawns and the column is not on the edge
                if (sum(1 for row in range(8) if self.board[row][col-1] == 'wP') == 0 and
                    sum(1 for row in range(8) if self.board[row][col+1] == 'wP') == 0):  # Check if there are no White pawns in adjacent columns
                    score -= 15  # Penalize White for having an isolated pawn
            if black_pawns > 0 and col > 0 and col < 7:  # Check if there are Black pawns and the column is not on the edge
                if (sum(1 for row in range(8) if self.board[row][col-1] == 'bP') == 0 and
                    sum(1 for row in range(8) if self.board[row][col+1] == 'bP') == 0):  # Check if there are no Black pawns in adjacent columns
                    score += 15  # Penalize Black for having an isolated pawn
        return score  # Return the pawn structure score

    def get_all_legal_moves(self, color=None, debug=False):
        color = color or self.turn  # Default to current turn if no color specified
        moves = []

        # Iterate over the board to find pieces of the specified player
        for row in range(8):
            for col in range(8):
                piece = self.board[row][col]
                if piece != "." and piece[0] == color:
                    piece_moves = self.get_piece_moves(row, col)
                    if debug:
                        print(f"Generating moves for {piece} at {self._coord_to_notation(row, col)}: {[self.to_algebraic(m) for m in piece_moves]}")
                    for move in piece_moves:
                        if self.is_legal_move(move.start, move.end):
                            moves.append(move)
                        elif debug:
                            print(f"Move {self.to_algebraic(move)} rejected: leaves {color} in check")

        # Sort moves: prioritize captures and checks
        def move_priority(move):
            is_capture = move.captured_piece is not None
            self.make_move(move, searching=True)
            opponent = "b" if self.turn == "w" else "w"
            puts_in_check = self.is_in_check(opponent)
            self.undo_move(searching=True)
            return (is_capture, puts_in_check)

        moves.sort(key=move_priority, reverse=True)
        if debug:
            print(f"Total legal moves generated: {len(moves)}, Sorted moves: {[self.to_algebraic(m) for m in moves]}")
        
        if not moves and debug:
            print(f"No legal moves found for {color}!")

        return moves

    def _coord_to_notation(self, row, col):
        """Helper to convert (row, col) to algebraic position (e.g., (6, 4) -> e2)."""
        files = "abcdefgh"
        return f"{files[col]}{8 - row}"

    def alpha_beta(self, depth, alpha, beta, maximizing_player, debug=False):
        if depth == 0 or self.is_game_over()[0]:
            score = self.evaluate_position()
            if debug:
                print(f"Depth {depth}, Eval Score: {score}, Maximizing: {maximizing_player}")
            return score

        # Generate moves for the correct player
        moves = self.get_all_legal_moves("w" if maximizing_player else "b", debug)
        if debug:
            print(f"Depth {depth}, Generated {len(moves)} moves for {'w' if maximizing_player else 'b'}: {[self.to_algebraic(m) for m in moves]}")

        if not moves:
            score = self.evaluate_position()
            if debug:
                print(f"Depth {depth}, No moves, Eval Score: {score}, Maximizing: {maximizing_player}")
            return score

        if maximizing_player:
            max_eval = float('-inf')
            for move in moves:
                self.make_move(move, searching=True)
                self.moves_evaluated += 1  # Increment counter here
                eval_score = self.alpha_beta(depth - 1, alpha, beta, False, debug)
                self.undo_move(searching=True)
                if debug:
                    print(f"Depth {depth}, Move: {self.to_algebraic(move)}, Eval Score: {eval_score}, Maximizing: True")
                max_eval = max(max_eval, eval_score)
                alpha = max(alpha, eval_score)
                if beta <= alpha:
                    if debug:
                        print(f"Pruning at depth {depth}, Move: {self.to_algebraic(move)}")
                    break
            return max_eval
        else:
            min_eval = float('inf')
            for move in moves:
                self.make_move(move, searching=True)
                self.moves_evaluated += 1  # Increment counter here
                eval_score = self.alpha_beta(depth - 1, alpha, beta, True, debug)
                self.undo_move(searching=True)
                if debug:
                    print(f"Depth {depth}, Move: {self.to_algebraic(move)}, Eval Score: {eval_score}, Maximizing: False")
                min_eval = min(min_eval, eval_score)
                beta = min(beta, eval_score)
                if beta <= alpha:
                    if debug:
                        print(f"Pruning at depth {depth}, Move: {self.to_algebraic(move)}")
                    break
            return min_eval

    def get_best_move(self, depth, debug=False):
        self.moves_evaluated = 0  # Reset counter before search
        if debug:
            print(f"Starting AI search for {self.turn} at depth {depth}")
        moves = self.get_all_legal_moves(self.turn, debug)  # Pass current turn
        if debug:
            print(f"Generated {len(moves)} moves for {self.turn}: {[self.to_algebraic(m) for m in moves]}")

        if not moves:
            if debug:
                print("No legal moves found!")
            return None

        best_score = float('-inf') if self.turn == "w" else float('inf')
        best_move = moves[0]  # Default to first legal move
        alpha = float('-inf')
        beta = float('inf')

        for move in moves:
            self.make_move(move, searching=True)
            if self.promotion_pending:
                self.complete_promotion('Q')
            self.moves_evaluated += 1  # Count top-level moves
            score = self.alpha_beta(depth - 1, alpha, beta, self.turn == "b", debug)
            self.undo_move(searching=True)
            if debug:
                print(f"Top-level Move: {self.to_algebraic(move)} ({move.piece} from {move.start} to {move.end}), Eval Score: {score}, Turn: {self.turn}")

            if self.turn == "w":  # Maximizing (White)
                if score > best_score:
                    best_score = score
                    best_move = move
                alpha = max(alpha, score)
            else:  # Minimizing (Black)
                if score < best_score:
                    best_score = score
                    best_move = move
                beta = min(beta, score)

            if beta <= alpha:
                if debug:
                    print(f"Pruning at move {self.to_algebraic(move)}")
                break

        print(f"AI evaluated {self.moves_evaluated} moves for {self.turn} at depth {depth}")
        if debug:
            print(f"AI selected move: {self.to_algebraic(best_move)}, Score: {best_score}")
        return best_move

    def get_piece_moves(self, row, col):
        piece = self.board[row][col]  # Get the piece at the given position
        if piece == ".":  # Check if the square is empty
            return []  # Return an empty list (no moves for an empty square)

        color, piece_type = piece[0], piece[1]  # Extract the piece's color and type
        moves = []  # Initialize an empty list to store the piece's moves

        if piece_type == "P":
            direction = -1 if color == "w" else 1  # White moves up (row decreases), Black down (row increases)
            start_row = 6 if color == "w" else 1   # White starts at row 6 (rank 2), Black at row 1 (rank 7)
            if 0 <= row + direction < 8:  # Check if one step forward is within the board
                if self.board[row + direction][col] == ".":  # Check if one step forward is empty
                    moves.append(Move((row, col), (row + direction, col), piece))
                # Check for two-square move from starting row
                    if row == start_row and 0 <= row + 2 * direction < 8 and self.board[row + direction][col] == "." and self.board[row + 2 * direction][col] == ".":
                        moves.append(Move((row, col), (row + 2 * direction, col), piece))
                # Captures to the left
                if col > 0 and self.board[row + direction][col - 1] != "." and self.board[row + direction][col - 1][0] != color:
                    moves.append(Move((row, col), (row + direction, col - 1), piece, self.board[row + direction][col - 1]))
                # Captures to the right
                if col < 7 and self.board[row + direction][col + 1] != "." and self.board[row + direction][col + 1][0] != color:
                    moves.append(Move((row, col), (row + direction, col + 1), piece, self.board[row + direction][col + 1]))
                # En passant
                if self.en_passant_target and color == self.turn:
                    ep_row, ep_col = self.en_passant_target
                    if row + direction == ep_row and abs(col - ep_col) == 1:
                        moves.append(Move((row, col), (ep_row, ep_col), piece, self.board[ep_row - direction][ep_col]))

        elif piece_type == "N":  # Check if the piece is a knight
            knight_moves = [
                (-2, -1), (-2, 1), (-1, -2), (-1, 2),
                (1, -2), (1, 2), (2, -1), (2, 1)
            ]  # Define the knight's possible moves (L-shape)
            for dr, dc in knight_moves:  # Loop through each possible knight move
                r, c = row + dr, col + dc  # Calculate the destination square
                if 0 <= r < 8 and 0 <= c < 8:  # Check if the destination is within the board
                    if self.board[r][c] == "." or self.board[r][c][0] != color:  # Check if the destination is empty or has an opponent's piece
                        moves.append(Move((row, col), (r, c), piece, self.board[r][c] if self.board[r][c] != "." else None))  # Add the move (with captured piece if applicable)

        elif piece_type == "B":  # Check if the piece is a bishop
            directions = [(-1, -1), (-1, 1), (1, -1), (1, 1)]  # Define the bishop's diagonal directions
            for dr, dc in directions:  # Loop through each diagonal direction
                r, c = row + dr, col + dc  # Calculate the next square in the direction
                while 0 <= r < 8 and 0 <= c < 8:  # Continue while the square is within the board
                    if self.board[r][c] == ".":  # Check if the square is empty
                        moves.append(Move((row, col), (r, c), piece))  # Add a move to the empty square
                    elif self.board[r][c][0] != color:  # Check if the square has an opponent's piece
                        moves.append(Move((row, col), (r, c), piece, self.board[r][c]))  # Add a capture move
                        break  # Stop after capturing an opponent's piece
                    else:  # If the square has a friendly piece
                        break  # Stop (cannot move through friendly pieces)
                    r += dr  # Move to the next square in the direction
                    c += dc  # Move to the next square in the direction

        elif piece_type == "R":  # Check if the piece is a rook
            directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Define the rook's straight directions (up, down, left, right)
            for dr, dc in directions:  # Loop through each straight direction
                r, c = row + dr, col + dc  # Calculate the next square in the direction
                while 0 <= r < 8 and 0 <= c < 8:  # Continue while the square is within the board
                    if self.board[r][c] == ".":  # Check if the square is empty
                        moves.append(Move((row, col), (r, c), piece))  # Add a move to the empty square
                    elif self.board[r][c][0] != color:  # Check if the square has an opponent's piece
                        moves.append(Move((row, col), (r, c), piece, self.board[r][c]))  # Add a capture move
                        break  # Stop after capturing an opponent's piece
                    else:  # If the square has a friendly piece
                        break  # Stop (cannot move through friendly pieces)
                    r += dr  # Move to the next square in the direction
                    c += dc  # Move to the next square in the direction

        elif piece_type == "Q":  # Check if the piece is a queen
            directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]  # Define the queen's directions (straight and diagonal)
            for dr, dc in directions:  # Loop through each direction
                r, c = row + dr, col + dc  # Calculate the next square in the direction
                while 0 <= r < 8 and 0 <= c < 8:  # Continue while the square is within the board
                    if self.board[r][c] == ".":  # Check if the square is empty
                        moves.append(Move((row, col), (r, c), piece))  # Add a move to the empty square
                    elif self.board[r][c][0] != color:  # Check if the square has an opponent's piece
                        moves.append(Move((row, col), (r, c), piece, self.board[r][c]))  # Add a capture move
                        break  # Stop after capturing an opponent's piece
                    else:  # If the square has a friendly piece
                        break  # Stop (cannot move through friendly pieces)
                    r += dr  # Move to the next square in the direction
                    c += dc  # Move to the next square in the direction

        elif piece_type == "K":
            # Standard king moves
            directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
            for dr, dc in directions:
                r, c = row + dr, col + dc
                if 0 <= r < 8 and 0 <= c < 8:
                    if self.board[r][c] == "." or self.board[r][c][0] != color:
                        moves.append(Move((row, col), (r, c), piece, self.board[r][c] if self.board[r][c] != "." else None))

            # Castling for White (row 7)
            if color == "w" and (row, col) == (7, 4) and (7, 4) not in self.moved_pieces:
                # Kingside
                if (7, 7) not in self.moved_pieces and all(self.board[7][c] == "." for c in range(5, 7)):
                    moves.append(Move((7, 4), (7, 6), piece, is_castling=True))
                # Queenside
                if (7, 0) not in self.moved_pieces and all(self.board[7][c] == "." for c in range(1, 4)):
                    moves.append(Move((7, 4), (7, 2), piece, is_castling=True))

            # Castling for Black (row 0)
            if color == "b" and (row, col) == (0, 4) and (0, 4) not in self.moved_pieces:
                # Kingside
                if (0, 7) not in self.moved_pieces and all(self.board[0][c] == "." for c in range(5, 7)):
                    moves.append(Move((0, 4), (0, 6), piece, is_castling=True))
                # Queenside
                if (0, 0) not in self.moved_pieces and all(self.board[0][c] == "." for c in range(1, 4)):
                    moves.append(Move((0, 4), (0, 2), piece, is_castling=True))

        return moves

    def is_in_check(self, color):
        king_pos = None  # Initialize the king's position as None
        for row in range(8):  # Loop through each row of the board
            for col in range(8):  # Loop through each column of the board
                piece = self.board[row][col]  # Get the piece at the current position
                if piece != "." and piece[0] == color and piece[1] == "K":  # Check if the piece is the king of the given color
                    king_pos = (row, col)  # Set the king's position
                    break  # Exit the column loop
            if king_pos:  # Check if the king was found
                break  # Exit the row loop

        if not king_pos:  # Check if the king was not found
            return False  # Return False (no king, no check)

        opponent = "b" if color == "w" else "w"  # Determine the opponent's color
        for row in range(8):  # Loop through each row of the board
            for col in range(8):  # Loop through each column of the board
                piece = self.board[row][col]  # Get the piece at the current position
                if piece != "." and piece[0] == opponent:  # Check if the piece belongs to the opponent
                    moves = self.get_piece_moves(row, col)  # Get all possible moves for the opponent's piece
                    for move in moves:  # Loop through each possible move
                        if move.end == king_pos:  # Check if the move attacks the king
                            return True  # Return True (king is in check)
        return False  # Return False (king is not in check)

    def is_legal_move(self, start, end):
        moves = self.get_piece_moves(start[0], start[1])
        move = None
        for m in moves:
            if m.end == end:
                move = m
                break
        if not move:
            return False

        # Simulate the move
        self.make_move(move, searching=True)
        in_check = self.is_in_check(self.turn)
        self.undo_move(searching=True)

        if in_check:  # Move leaves king in check
            return False

        # Additional castling checks
        if move.is_castling:
            king_row = start[0]
            color = move.piece[0]
            if color == "w" and king_row != 7:
                return False
            if color == "b" and king_row != 0:
                return False
            # Check squares king passes through
            if end[1] == 6:  # Kingside
                for c in [4, 5, 6]:  # e, f, g
                    self.board[king_row][4] = "."
                    self.board[king_row][c] = move.piece
                    if self.is_in_check(color):
                        self.board[king_row][c] = "."
                        self.board[king_row][4] = move.piece
                        return False
                    self.board[king_row][c] = "."
                    self.board[king_row][4] = move.piece
            elif end[1] == 2:  # Queenside
                for c in [4, 3, 2]:  # e, d, c
                    self.board[king_row][4] = "."
                    self.board[king_row][c] = move.piece
                    if self.is_in_check(color):
                        self.board[king_row][c] = "."
                        self.board[king_row][4] = move.piece
                        return False
                    self.board[king_row][c] = "."
                    self.board[king_row][4] = move.piece

        return True

    def make_move(self, move, move_str=None, searching=False):
        x, y = move.start
        ex, ey = move.end
        piece = self.board[x][y]
        if piece == ".":
            if not searching:
                print("No piece at that position.")
            return
        if not searching and ((piece[0] == "w" and self.turn == "b") or (piece[0] == "b" and self.turn == "w")):
            print("Not your turn!")
            return
        if move.captured_piece and piece[0] == move.captured_piece[0]:
            if not searching:
                print("You cannot capture your own piece!")
            return

        # Skip legality check if searching (already validated by is_legal_move)
        if not searching and not self.is_legal_move((x, y), (ex, ey)):
            print("Illegal move: leaves king in check!")
            return

        # Save state for undo
        self.save_state()

        # Update halfmove clock
        if piece[1] == "P" or move.captured_piece:
            self.halfmove_clock = 0
        else:
            self.halfmove_clock += 1

        # Handle castling
        if move.is_castling:
            if ey == 6:  # Kingside
                self.board[x][5] = self.board[x][7]
                self.board[x][7] = "."
                self.moved_pieces.add((x, 7))
            elif ey == 2:  # Queenside
                self.board[x][3] = self.board[x][0]
                self.board[x][0] = "."
                self.moved_pieces.add((x, 0))

        # Handle en passant
        elif piece[1] == "P" and move.end == self.en_passant_target:
            direction = -1 if piece[0] == "w" else 1
            capture_row = ex - direction
            capture_col = ey
            self.board[capture_row][capture_col] = "."
            if not searching:
                print(f"En passant capture: Removed pawn at ({capture_row}, {capture_col})")

        # Handle promotion
        if piece[1] == "P" and (ex == 0 or ex == 7):
            self.promotion_pending = move
            self.board[ex][ey] = piece
            self.board[x][y] = "."
            if not searching:
                move_str = self.to_algebraic(move)
                if self.turn == "w":
                    self.move_history.append(f"{self.move_number}. {move_str}")
                else:
                    self.move_history[-1] += f" {move_str}"
            return

        # Standard move
        self.board[ex][ey] = piece
        self.board[x][y] = "."
        if move.captured_piece and move.end != self.en_passant_target:
            if not searching:
                print(f"Captured {move.captured_piece} at ({ex}, {ey})")
        self.moved_pieces.add((x, y))
        self.previous_move = move

        # Update move history with the provided move_str
        if not searching:
            if move_str is None:  # Fallback if no string provided (e.g., during search)
                move_str = self.to_algebraic(move)
            # **CHANGED: Removed internal to_algebraic call and check logic**
            # Move the board update before history to ensure state is consistent
            self.board[ex][ey] = piece
            self.board[x][y] = "."
            opponent = "b" if self.turn == "w" else "w"
            if self.is_in_check(opponent):  # Add check symbol after move is applied
                move_str += "+"
            if self.turn == "w":
                self.move_history.append(f"{self.move_number}. {move_str}")
            else:
                self.move_history[-1] += f" {move_str}"
                self.move_number += 1
        else:
            # For searching, apply move without history update
            self.board[ex][ey] = piece
            self.board[x][y] = "."

        # Set or clear en passant target
        if piece[1] == "P" and abs(x - ex) == 2:
            self.en_passant_target = ((x + ex) // 2, y)
        else:
            self.en_passant_target = None

        # Switch turn
        if not searching:
            self.turn = "b" if self.turn == "w" else "w"
            position_key = self.get_position_key()
            self.position_history[position_key] = self.position_history.get(position_key, 0) + 1
            print(f"Move executed: {self.to_algebraic(move)}. Now it's {self.turn}'s turn.")

    def complete_promotion(self, choice):
        if not self.promotion_pending:  # Check if there is no promotion pending
            return  # Exit the method
        move = self.promotion_pending  # Get the pending promotion move
        x, y = move.start  # Get the starting row and column
        ex, ey = move.end  # Get the ending row and column
        if not (0 <= ex < 8 and 0 <= ey < 8):
            print(f"Error: Invalid promotion coordinates ({ex}, {ey})")
            return
        piece = move.piece  # Get the piece (pawn) being promoted
        new_piece = f"{self.turn}{choice}"  # Create the new piece (e.g., "wQ" for a white queen)
        self.board[ex][ey] = new_piece  # Replace the pawn with the promoted piece
        self.moved_pieces.add((x, y))  # Mark the pawn as moved
        self.previous_move = move  # Store the move as the previous move

        self.halfmove_clock = 0  # Reset the halfmove clock (promotion counts as a significant move)

        if not self.searching:  # Check if not in search mode
            if self.turn == "w":  # Check if it's White's turn
                base_move = self.move_history[-1].split(" ")[1]  # Get the base move notation (before promotion)
            else:  # If it's Black's turn
                base_move = self.move_history[-1].split(" ")[2]  # Get the base move notation (before promotion)

            move.promotion = choice  # Set the promotion choice in the move object
            move_str = self.to_algebraic(move)  # Convert the move to algebraic notation with the promotion
            opponent = "b" if self.turn == "w" else "w"  # Determine the opponent's color
            if self.is_in_check(opponent):  # Check if the move puts the opponent in check
                move_str += "+"  # Add "+" to indicate check

            print(f"Promoting to {new_piece}, move: {move_str}")  # Print a message indicating the promotion
            if self.turn == "w":  # Check if it's White's turn
                self.move_history[-1] = f"{self.move_number}. {move_str}"  # Update the move history with the promotion
            else:  # If it's Black's turn
                parts = self.move_history[-1].split(" ")  # Split the current move pair
                self.move_history[-1] = f"{parts[0]} {parts[1]} {move_str}"  # Update the move pair with the promotion
                self.move_number += 1  # Increment the move number after Black's move

            position_key = self.get_position_key()  # Get the current position key
            self.position_history[position_key] = self.position_history.get(position_key, 0) + 1  # Increment the position's occurrence count

        self.turn = "b" if self.turn == "w" else "w"  # Switch the turn to the other player
        self.promotion_pending = None  # Clear the promotion pending flag
        if not self.searching:  # Check if not in search mode
            print(f"Promotion to {new_piece}. Now it's {self.turn}'s turn.")  # Print a message indicating the promotion is complete

    def undo_move(self, searching=False):
        if not self.state_history:  # Check if there are no states to undo
            if not searching:  # Only print if not simulating
                print("No moves to undo!")
            return

        # Restore state from history
        state = self.state_history.pop()
        self.board = [row[:] for row in state["board"]]  # Deep copy of the board
        self.turn = state["turn"]
        self.moved_pieces = state["moved_pieces"].copy()  # Copy to avoid modifying saved state
        self.previous_move = state["previous_move"]
        self.en_passant_target = state["en_passant_target"]
        self.halfmove_clock = state["halfmove_clock"]
        self.promotion_pending = state.get("promotion_pending", None)  # Restore promotion if present

        # Update move history and position history only if not searching
        if not searching:
            if self.move_history:  # Remove the last move
                self.move_history.pop()
            self.move_number = state["move_number"]  # Restore move number
            position_key = self.get_position_key()
            self.position_history[position_key] = self.position_history.get(position_key, 0) - 1
            if self.position_history[position_key] <= 0:
                del self.position_history[position_key]

    def is_game_over(self):
        # Check for 50-move rule
        if self.halfmove_clock >= 100:  # Check if 100 halfmoves have passed (50 full moves)
            return (True, "Game drawn by 50-move rule!")  # Return that the game is over by the 50-move rule

        # Check for threefold repetition
        position_key = self.get_position_key()  # Get the current position key
        if self.position_history.get(position_key, 0) >= 3:  # Check if the position has occurred 3 or more times
            return (True, "Game drawn by threefold repetition!")  # Return that the game is over by threefold repetition

        # Check if the current player has any legal moves
        has_legal_moves = False  # Initialize a flag to track if there are legal moves
        in_check = self.is_in_check(self.turn)  # Check if the current player is in check

        # Iterate through all pieces of the current player
        for row in range(8):  # Loop through each row of the board
            for col in range(8):  # Loop through each column of the board
                piece = self.board[row][col]  # Get the piece at the current position
                if piece != "." and piece[0] == self.turn:  # Check if the piece belongs to the current player
                    moves = self.get_piece_moves(row, col)  # Get all possible moves for the piece
                    for move in moves:  # Loop through each possible move
                        if self.is_legal_move(move.start, move.end):  # Check if the move is legal
                            has_legal_moves = True  # Set the flag to True
                            break  # Exit the move loop
                if has_legal_moves:  # Check if a legal move was found
                    break  # Exit the column loop
            if has_legal_moves:  # Check if a legal move was found
                break  # Exit the row loop

        # Determine the game result
        if not has_legal_moves:  # Check if there are no legal moves
            if in_check:  # Check if the player is in check
                winner = "Black" if self.turn == "w" else "White"  # Determine the winner
                return (True, f"{winner} wins by checkmate!")  # Return that the game is over by checkmate
            else:
                return (True, "Game drawn by stalemate!")  # Return that the game is over by stalemate

        return (False, "")  # Return that the game is not over

    def resign(self):
        winner = "Black" if self.turn == "w" else "White"  # Determine the winner based on who resigned
        result = f"{winner} wins by resignation!"  # Create the result message
        return (True, result)  # Return that the game is over by resignation

    def draw(self):
        return (True, "Game drawn by agreement!")  # Return that the game is over by mutual agreement (draw)

    def generate_pgn(self, result):
        pgn = f'[Event "Casual Game"]\n'  # Add the event tag to the PGN
        pgn += f'[Site "Local Computer"]\n'  # Add the site tag to the PGN
        pgn += f'[Date "{datetime.datetime.now().strftime("%Y.%m.%d")}"]\n'  # Add the date tag to the PGN (current date)
        pgn += f'[Round "1"]\n'  # Add the round tag to the PGN
        pgn += f'[White "Player1"]\n'  # Add the White player tag to the PGN
        pgn += f'[Black "Player2"]\n'  # Add the Black player tag to the PGN
        pgn += f'[Result "{result}"]\n\n'  # Add the result tag to the PGN (e.g., "1-0")

        pgn += " ".join(self.move_history)  # Join with single spaces
        pgn += " " + result + "\n"
        return pgn
