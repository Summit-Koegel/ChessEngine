import pygame  # Import the Pygame library for creating the game window and handling graphics
import sys  # Import the sys module for system-specific functions like exiting the program
import os  # Import the os module for interacting with the operating system (e.g., file paths)
import datetime  # Import the datetime module to generate timestamps for saving game files
from ChessEngine import ChessEngine, Move  # type: ignore  # Import ChessEngine and Move classes from ChessEngine module (type: ignore suppresses type checking warnings)

#Comment
class ChessGame:
    def __init__(self):
        self.engine = ChessEngine()  # Create an instance of ChessEngine to manage game logic
        print("Initial board state:")  # Print a message indicating the initial board state will be displayed
        for row in self.engine.board:  # Loop through each row of the chessboard
            print(row)  # Print the current row to show the initial board setup
        self.screen = None  # Initialize the screen variable as None (will be set later by Pygame)
        self.square_size = 100  # Set the size of each chessboard square to 100 pixels
        self.piece_images = {}  # Initialize an empty dictionary to store chess piece images
        self.load_piece_images()  # Call the method to load chess piece images into the dictionary
        self.flipped = False 
        self.highlighted_square = None  # Initialize the highlighted square as None (for user interaction)
        self.selected_square = None  # Initialize the selected square as None (for piece selection)
        self.valid_moves = []  # Initialize an empty list to store valid moves for the selected piece
        self.ai_enabled = False  # Set the AI mode to off by default (flag to toggle AI)
        self.redraw_needed = True  # Add a flag to track if redraw is needed

    def load_piece_images(self):
        pieces = ["P", "N", "B", "R", "Q", "K"]  # Define a list of piece types (Pawn, Knight, Bishop, Rook, Queen, King)
        colors = ["w", "b"]  # Define a list of piece colors (white, black)
        script_dir = os.path.dirname(os.path.abspath(__file__))  # Get the directory of the current script file
        for color in colors:  # Loop through each color (white and black)
            for piece in pieces:  # Loop through each piece type
                image_path = os.path.join(script_dir, "Images", f"{color}{piece}.png")  # Construct the file path for the piece image
                try:
                    self.piece_images[f"{color}{piece}"] = pygame.transform.scale(
                        pygame.image.load(image_path), (self.square_size, self.square_size)
                    )  # Load and scale the piece image to the square size, then store it in the dictionary
                except FileNotFoundError:
                    print(f"Error: Could not load {image_path}. Ensure 'Images' folder is in {script_dir}.")  # Print an error if the image file is not found
                    raise  # Raise the exception to stop the program if the image cannot be loaded

    def draw_board(self, screen):
        cream = (240, 217, 181)  # Define the cream color for light squares
        light_brown = (181, 136, 99)  # Define the light brown color for dark squares
        highlight_cream = (255, 255, 230)  # Define the highlight color for light squares
        highlight_brown = (230, 180, 140)  # Define the highlight color for dark squares
        check_red = (255, 100, 100)  # Define the red color to highlight a king in check

        for row in range(8):  # Loop through each row of the chessboard (0 to 7)
            for col in range(8):  # Loop through each column of the chessboard (0 to 7)
                display_row = 7 - row if self.flipped else row  # Adjust the row for display based on whether the board is flipped
                display_col = 7- col if self.flipped else col 
                x = display_col * self.square_size  # Calculate the x-coordinate of the square on the screen
                y = display_row * self.square_size  # Calculate the y-coordinate of the square on the screen
                color = cream if (row + col) % 2 == 0 else light_brown  # Set the square color (cream for light, light brown for dark)
                if self.highlighted_square == (row, col):  # Check if the current square is highlighted
                    color = highlight_cream if (row + col) % 2 == 0 else highlight_brown  # Use highlight color if the square is highlighted
                elif self.engine.board[row][col] == f"{self.engine.turn}K" and self.engine.is_in_check(self.engine.turn):  # Check if the square has the current player's king in check
                    color = check_red  # Use red to highlight the king in check
                pygame.draw.rect(screen, color, (x, y, self.square_size, self.square_size))  # Draw the square on the screen with the chosen color

        # Draw pieces
        for row in range(8):  # Loop through each row of the chessboard
            for col in range(8):  # Loop through each column of the chessboard
                piece = self.engine.board[row][col]  # Get the piece at the current position
                if piece != ".":  # Check if there is a piece (not an empty square)
                    flipped_row = 7 - row if self.flipped else row  # Adjust the row for display if the board is flipped
                    flipped_col = 7 - col if self.flipped else col # Keep the column as is
                    screen.blit(
                        self.piece_images[piece],
                        (flipped_col * self.square_size, flipped_row * self.square_size),
                    )  # Draw the piece image on the screen at the adjusted position

    def draw_sidebar(self, screen):
        sidebar_x = 800  # Set the x-coordinate of the sidebar (right side of the screen)
        sidebar_width = 200  # Set the width of the sidebar
        sidebar_height = 800  # Set the height of the sidebar (full height of the screen)
        pygame.draw.rect(screen, (200, 200, 200), (sidebar_x, 0, sidebar_width, sidebar_height))  # Draw the sidebar background as a gray rectangle

        font = pygame.font.SysFont(None, 30)  # Create a font object for rendering text (size 30)
        # Flip Button
        flip_text = font.render("Flip Board", True, (0, 0, 0))  # Render the "Flip Board" text in black
        flip_rect = flip_text.get_rect(center=(sidebar_x + sidebar_width // 2, 50))  # Position the flip button text in the center of the sidebar at y=50
        pygame.draw.rect(screen, (150, 150, 150), (sidebar_x + 50, 30, 100, 40))  # Draw a gray button background for the flip button
        screen.blit(flip_text, flip_rect)  # Draw the "Flip Board" text on the button

        # Undo Button
        undo_text = font.render("Undo", True, (0, 0, 0))  # Render the "Undo" text in black
        undo_rect = undo_text.get_rect(center=(sidebar_x + sidebar_width // 2, 100))  # Position the undo button text at y=100
        pygame.draw.rect(screen, (150, 150, 150), (sidebar_x + 50, 80, 100, 40))  # Draw a gray button background for the undo button
        screen.blit(undo_text, undo_rect)  # Draw the "Undo" text on the button

        # Print FEN Button
        fen_text = font.render("Print FEN", True, (0, 0, 0))  # Render the "Print FEN" text in black
        fen_rect = fen_text.get_rect(center=(sidebar_x + sidebar_width // 2, 150))  # Position the FEN button text at y=150
        pygame.draw.rect(screen, (150, 150, 150), (sidebar_x + 50, 130, 100, 40))  # Draw a gray button background for the FEN button
        screen.blit(fen_text, fen_rect)  # Draw the "Print FEN" text on the button

        # Resign Button
        resign_text = font.render("Resign", True, (0, 0, 0))  # Render the "Resign" text in black
        resign_rect = resign_text.get_rect(center=(sidebar_x + sidebar_width // 2, 200))  # Position the resign button text at y=200
        pygame.draw.rect(screen, (150, 150, 150), (sidebar_x + 50, 180, 100, 40))  # Draw a gray button background for the resign button
        screen.blit(resign_text, resign_rect)  # Draw the "Resign" text on the button

        # Draw Button
        draw_text = font.render("Draw", True, (0, 0, 0))  # Render the "Draw" text in black
        draw_rect = draw_text.get_rect(center=(sidebar_x + sidebar_width // 2, 250))  # Position the draw button text at y=250
        pygame.draw.rect(screen, (150, 150, 150), (sidebar_x + 50, 230, 100, 40))  # Draw a gray button background for the draw button
        screen.blit(draw_text, draw_rect)  # Draw the "Draw" text on the button

        # AI Button
        ai_text = font.render("Toggle AI", True, (0, 0, 0))  # Render the "Toggle AI" text in black
        ai_rect = ai_text.get_rect(center=(sidebar_x + sidebar_width // 2, 300))  # Position the AI button text at y=300
        pygame.draw.rect(screen, (150, 150, 150), (sidebar_x + 50, 280, 100, 40))  # Draw a gray button background for the AI button
        screen.blit(ai_text, ai_rect)  # Draw the "Toggle AI" text on the button

        # Promotion options (if pending)
        if self.engine.promotion_pending:  # Check if a pawn promotion is pending
            buttons = [("Q", 350), ("R", 400), ("N", 450), ("B", 500)]  # Define promotion options (Queen, Rook, Knight, Bishop) and their y-positions
            for piece, y_pos in buttons:  # Loop through each promotion option
                text = font.render(piece, True, (0, 0, 0))  # Render the piece symbol in black
                rect = text.get_rect(center=(sidebar_x + sidebar_width // 2, y_pos))  # Position the promotion button text
                pygame.draw.rect(screen, (255, 255, 255), (sidebar_x + 50, y_pos - 20, 100, 40))  # Draw a white button background for the promotion option
                screen.blit(text, rect)  # Draw the piece symbol on the button

        log_y = 550  # Set the starting y-position for the move history log (adjusted for button space)
        for i, move in enumerate(self.engine.move_history[-10:]):  # Loop through the last 10 moves in the move history
            text = font.render(move, True, (0, 0, 0))  # Render the move text in black
            screen.blit(text, (sidebar_x + 10, log_y + i * 30))  # Draw the move text on the sidebar, spaced 30 pixels apart

    def get_square_from_mouse_pos(self, pos):
        x, y = pos
        row = y // self.square_size
        col = x // self.square_size
        if self.flipped:  # When flipped, match screen (top=0, bottom=7)
            row = 7 - row
            col = 7 - col
        return row, col

    def start_game(self):
        pygame.init()  # Initialize Pygame to set up the game environment
        self.screen = pygame.display.set_mode((1000, 800))  # Create a game window of size 1000x800 pixels
        pygame.display.set_caption("Chess")  # Set the window title to "Chess"
        self.selected_square = None  # Reset the selected square to None
        self.valid_moves = []  # Reset the valid moves list to empty
        running = True  # Set the game loop flag to True to keep the game running
        human_color = 'w'  # Set the human player to control White
        ai_color = 'b'  # Set the AI to control Black

        while running:  # Start the main game loop
            # Process events first
            for event in pygame.event.get():  # Loop through all Pygame events (e.g., mouse clicks, key presses)
                if event.type == pygame.QUIT:  # Check if the user closed the window
                    running = False  # Set the running flag to False to exit the game loop
                elif event.type == pygame.MOUSEBUTTONDOWN:  # Check if the user clicked the mouse
                    mouse_pos = pygame.mouse.get_pos()  # Get the mouse position (x, y coordinates)
                    clicked_piece = None  # Initialize the clicked piece as None
                    if mouse_pos[0] >= 800:  # Check if the click was in the sidebar (x >= 800)
                        if 30 <= mouse_pos[1] <= 70:  # Check if the click was on the "Flip Board" button
                            self.flipped = not self.flipped  # Toggle the board perspective (flipped or not)
                            print(f"Flipped: {self.flipped}")  # Print the new flipped state
                        elif 80 <= mouse_pos[1] <= 120:  # Check if the click was on the "Undo" button
                            self.engine.undo_move()  # Call the engine to undo the last move
                        elif 130 <= mouse_pos[1] <= 170:  # Check if the click was on the "Print FEN" button
                            print(self.engine.get_fen())  # Print the current position in FEN notation
                        elif 180 <= mouse_pos[1] <= 220:  # Check if the click was on the "Resign" button
                            is_over, result = self.engine.resign()  # Call the engine to handle resignation
                            print(result)  # Print the result of the resignation (e.g., "White wins")
                            pgn_result = "1-0" if "White wins" in result else "0-1"  # Set the PGN result based on who wins
                            pgn = self.engine.generate_pgn(pgn_result)  # Generate the game record in PGN format
                            desktop_path = os.path.join(os.path.expanduser("~"), "Desktop", "GameArchive")  # Define the path to save the game on the desktop
                            os.makedirs(desktop_path, exist_ok=True)  # Create the GameArchive directory if it doesn't exist
                            timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")  # Generate a timestamp for the file name
                            file_path = os.path.join(desktop_path, f"game_{timestamp}.pgn")  # Construct the full file path for the PGN file
                            with open(file_path, "w", encoding="ascii") as f:
                                f.write(pgn.strip())  # Write the PGN data to the file
                            print(f"Game saved to {os.path.abspath(file_path)}")  # Print the location where the game was saved
                            font = pygame.font.SysFont(None, 48)  # Create a font object for displaying the result
                            text = font.render(result, True, (255, 0, 0))  # Render the result text in red
                            text_rect = text.get_rect(center=(400, 400))  # Center the text on the screen
                            self.screen.blit(text, text_rect)  # Draw the result text on the screen
                            pygame.display.flip()  # Update the display to show the result
                            pygame.time.wait(4000)  # Wait for 3 seconds to let the user see the result
                            running = False  # Exit the game loop
                        elif 230 <= mouse_pos[1] <= 270:  # Check if the click was on the "Draw" button
                            is_over, result = self.engine.draw()  # Call the engine to handle a draw
                            print(result)  # Print the result of the draw (e.g., "Game drawn")
                            pgn_result = "1/2-1/2"  # Set the PGN result for a draw
                            pgn = self.engine.generate_pgn(pgn_result)  # Generate the game record in PGN format
                            desktop_path = os.path.join(os.path.expanduser("~"), "Desktop", "GameArchive")  # Define the path to save the game
                            os.makedirs(desktop_path, exist_ok=True)  # Create the GameArchive directory if it doesn't exist
                            timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")  # Generate a timestamp for the file name
                            file_path = os.path.join(desktop_path, f"game_{timestamp}.pgn")  # Construct the full file path for the PGN file
                            with open(file_path, "w", encoding="ascii") as f:
                                f.write(pgn.strip()) # Write the PGN data to the file
                            print(f"Game saved to {os.path.abspath(file_path)}")  # Print the location where the game was saved
                            font = pygame.font.SysFont(None, 48)  # Create a font object for displaying the result
                            text = font.render(result, True, (255, 0, 0))  # Render the result text in red
                            text_rect = text.get_rect(center=(400, 400))  # Center the text on the screen
                            self.screen.blit(text, text_rect)  # Draw the result text on the screen
                            pygame.display.flip()  # Update the display to show the result
                            pygame.time.wait(4000)  # Wait for 3 seconds to let the user see the result
                            running = False  # Exit the game loop
                        elif 280 <= mouse_pos[1] <= 320:  # Check if the click was on the "Toggle AI" button
                            self.ai_enabled = not self.ai_enabled  # Toggle the AI mode on or off
                            print(f"AI enabled: {self.ai_enabled}")  # Print the new AI state
                        elif self.engine.promotion_pending:  # Check if a pawn promotion is pending
                            print(f"Promotion click at y={mouse_pos[1]}")  # Print the y-coordinate of the promotion click
                            if 330 <= mouse_pos[1] <= 370:  # Check if the click was on the "Queen" promotion button
                                print("Clicked Q")  # Print a confirmation message
                                self.engine.complete_promotion("Q")  # Promote the pawn to a Queen
                            elif 400 <= mouse_pos[1] <= 440:  # Check if the click was on the "Rook" promotion button
                                self.engine.complete_promotion("R")  # Promote the pawn to a Rook
                            elif 450 <= mouse_pos[1] <= 490:  # Check if the click was on the "Knight" promotion button
                                self.engine.complete_promotion("N")  # Promote the pawn to a Knight
                            elif 500 <= mouse_pos[1] <= 540:  # Check if the click was on the "Bishop" promotion button
                                self.engine.complete_promotion("B")  # Promote the pawn to a Bishop
                    else:  # If the click was on the chessboard (not the sidebar)
                        row, col = self.get_square_from_mouse_pos(mouse_pos)  # Convert the mouse position to board coordinates
                        try:
                            clicked_piece = self.engine.board[row][col]  # Get the piece at the clicked position
                            print(f"Selected square: ({row}, {col}), Piece: {clicked_piece}")  # Print the selected square and piece
                        except IndexError:
                            print(f"Invalid coordinates: ({row}, {col})")  # Print an error if the coordinates are out of bounds
                            clicked_piece = "."  # Set the clicked piece to empty if the coordinates are invalid

                       # If a piece is already selected
                        if self.selected_square and not self.engine.promotion_pending:
                            # Check if the clicked square has a piece of the current player's color
                            if clicked_piece != "." and clicked_piece[0] == self.engine.turn:
                                # Reselect the new piece
                                self.selected_square = (row, col)
                                self.valid_moves = self.engine.get_piece_moves(row, col)
                                print(f"Reselected piece. Valid moves: {[self.engine.to_algebraic(m) for m in self.valid_moves]}")
                                self.highlighted_square = (row, col)
                            else:
                                # Try to execute a move to the clicked square
                                move = None
                                for m in self.valid_moves:
                                    if m.end == (row, col):
                                        move = m
                                        break
                                if move:
                                    print(f"Found move: {self.engine.to_algebraic(move)} from {move.start} to {move.end}")
                                    if self.engine.is_legal_move(move.start, move.end):
                                        print(f"Executing move: {self.engine.to_algebraic(move)}")
                                        move_str = self.engine.to_algebraic(move)
                                        print(f"Executing move: {move_str}")
                                        self.engine.make_move(move, move_str)
                                        self.selected_square = None
                                        self.valid_moves = []
                                        self.highlighted_square = None
                                        ai_moved = False
                                        if self.ai_enabled and self.engine.turn == ai_color and not self.engine.is_game_over()[0] and not ai_moved:
                                            print(f"Triggering AI move for {self.engine.turn}")
                                            ai_move = self.engine.get_best_move(depth=4)
                                            if ai_move:
                                                print(f"AI chose {self.engine.to_algebraic(ai_move)}")
                                                ai_move_str = self.engine.to_algebraic(ai_move)
                                                print(f"AI chose {ai_move_str}")
                                                self.engine.make_move(ai_move, ai_move_str)
                                                if self.engine.promotion_pending:
                                                    self.engine.complete_promotion("Q")
                                                ai_moved = True
                                            else:
                                                print("AI failed to find a move")
                                    else:
                                        print(f"Move {self.engine.to_algebraic(move)} is illegal")
                                else:
                                    print(f"No valid move to ({row}, {col}) in {[(m.start, m.end) for m in self.valid_moves]}")
                                    # Deselect if clicking an empty square or opponent's piece
                                    self.selected_square = None
                                    self.valid_moves = []
                                    self.highlighted_square = None
                        else:
                            # No piece selected yet, select one
                            if clicked_piece != "." and clicked_piece[0] == self.engine.turn:
                                self.selected_square = (row, col)
                                self.valid_moves = self.engine.get_piece_moves(row, col)
                                print(f"Valid moves: {[self.engine.to_algebraic(m) for m in self.valid_moves]}")
                                self.highlighted_square = (row, col)
                            else:
                                self.selected_square = None
                                self.valid_moves = []
                                self.highlighted_square = None
                        self.redraw_needed = True

            if self.redraw_needed:
                self.screen.fill((255, 255, 255))
                self.draw_board(self.screen)
                if self.valid_moves:
                    for move in self.valid_moves:
                        row, col = move.end
                        if self.flipped:
                            row = 7 - row
                            col = 7 - col
                        x = col * self.square_size + self.square_size // 2
                        y = row * self.square_size + self.square_size // 2
                        pygame.draw.circle(self.screen, (12, 60, 5), (x, y), 10)
                self.draw_sidebar(self.screen)
                pygame.display.flip()
                self.redraw_needed = False  # Reset flag after drawing

            is_over, result = self.engine.is_game_over()
            if is_over:
                print("Move history:", self.engine.move_history)
                print("Game over result:", result)
                # Robustly determine PGN result
                if "checkmate" in result.lower():
                    pgn_result = "1-0" if self.engine.turn == "b" else "0-1"  # Winner is opponent of current turn
                else:
                    pgn_result = "1/2-1/2"  # Stalemate or draw
                pgn = self.engine.generate_pgn(pgn_result)
                print("Raw PGN:", repr(pgn))
                desktop_path = os.path.join(os.path.expanduser("~"), "Desktop", "GameArchive")
                os.makedirs(desktop_path, exist_ok=True)
                timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                file_path = os.path.join(desktop_path, f"game_{timestamp}.pgn")
                with open(file_path, "w", encoding="ascii") as f:
                    f.write(pgn.strip())
                print(f"Game saved to {os.path.abspath(file_path)}")
                font = pygame.font.SysFont(None, 48)
                text = font.render(result, True, (255, 0, 0))
                text_rect = text.get_rect(center=(400, 400))
                self.screen.blit(text, text_rect)
                pygame.display.flip()
                pygame.time.wait(4000)
                running = False

            pygame.display.flip()  # Update the entire display to show the current frame

        pygame.quit()  # Quit Pygame to clean up resources
        sys.exit()  # Exit the program

if __name__ == "__main__":
    chess_game = ChessGame()  # Create an instance of the ChessGame class
    chess_game.start_game()  # Start the game by calling the start_game method