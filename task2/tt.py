import pygame
import sys

# Initialize pygame
pygame.init()

# Constants
WIDTH, HEIGHT = 600, 700
GRID_SIZE = 3
CELL_SIZE = WIDTH // GRID_SIZE
LINE_WIDTH = 4
RADIUS = CELL_SIZE // 4
OFFSET = 40

# Colors
BG_COLOR = (250, 250, 250)
LINE_COLOR = (200, 200, 200)
X_COLOR = (60, 60, 60)
O_COLOR = (90, 160, 255)
WIN_COLOR = (50, 220, 100)
BUTTON_COLOR = (240, 240, 240)
BUTTON_BORDER = (200, 200, 200)

# Fonts
FONT = pygame.font.SysFont("Helvetica", 48)
INFO_FONT = pygame.font.SysFont("Helvetica", 32)

# Screen
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Tic-Tac-Toe AI")
clock = pygame.time.Clock()

# Game state
board = [' '] * 9
game_over = False
player_turn = True
winner = None
win_combo = None


def draw_grid():
    screen.fill(BG_COLOR)
    for i in range(1, GRID_SIZE):
        pygame.draw.line(screen, LINE_COLOR, (0, i * CELL_SIZE), (WIDTH, i * CELL_SIZE), LINE_WIDTH)
        pygame.draw.line(screen, LINE_COLOR, (i * CELL_SIZE, 0), (i * CELL_SIZE, WIDTH), LINE_WIDTH)


def draw_marks():
    for i in range(9):
        x = (i % GRID_SIZE) * CELL_SIZE
        y = (i // GRID_SIZE) * CELL_SIZE
        if board[i] == 'X':
            pygame.draw.line(screen, X_COLOR, (x + OFFSET, y + OFFSET),
                             (x + CELL_SIZE - OFFSET, y + CELL_SIZE - OFFSET), LINE_WIDTH * 2)
            pygame.draw.line(screen, X_COLOR, (x + CELL_SIZE - OFFSET, y + OFFSET),
                             (x + OFFSET, y + CELL_SIZE - OFFSET), LINE_WIDTH * 2)
        elif board[i] == 'O':
            center = (x + CELL_SIZE // 2, y + CELL_SIZE // 2)
            pygame.draw.circle(screen, O_COLOR, center, RADIUS, LINE_WIDTH * 2)


def get_empty_cells():
    return [i for i, val in enumerate(board) if val == ' ']


def check_winner(symbol):
    combos = [(0,1,2), (3,4,5), (6,7,8),
              (0,3,6), (1,4,7), (2,5,8),
              (0,4,8), (2,4,6)]
    for combo in combos:
        if all(board[i] == symbol for i in combo):
            return combo
    return None


def is_draw():
    return ' ' not in board


def minimax(is_max):
    if check_winner('O'):
        return 1
    if check_winner('X'):
        return -1
    if is_draw():
        return 0

    if is_max:
        best = -float('inf')
        for move in get_empty_cells():
            board[move] = 'O'
            best = max(best, minimax(False))
            board[move] = ' '
        return best
    else:
        best = float('inf')
        for move in get_empty_cells():
            board[move] = 'X'
            best = min(best, minimax(True))
            board[move] = ' '
        return best


def best_ai_move():
    best_score = -float('inf')
    best_move = None
    for move in get_empty_cells():
        board[move] = 'O'
        score = minimax(False)
        board[move] = ' '
        if score > best_score:
            best_score = score
            best_move = move
    return best_move


def draw_win_line(combo):
    x1, y1 = (combo[0] % 3) * CELL_SIZE + CELL_SIZE // 2, (combo[0] // 3) * CELL_SIZE + CELL_SIZE // 2
    x2, y2 = (combo[2] % 3) * CELL_SIZE + CELL_SIZE // 2, (combo[2] // 3) * CELL_SIZE + CELL_SIZE // 2
    pygame.draw.line(screen, WIN_COLOR, (x1, y1), (x2, y2), 10)


def draw_result():
    if winner:
        text = "You Win!" if winner == 'X' else "AI Wins!"
    else:
        text = "Draw!"
    label = INFO_FONT.render(text, True, (0, 0, 0))
    screen.blit(label, (WIDTH // 2 - label.get_width() // 2, HEIGHT - 80))


def reset():
    global board, game_over, player_turn, winner, win_combo
    board = [' '] * 9
    game_over = False
    player_turn = True
    winner = None
    win_combo = None


def draw_reset_button():
    rect = pygame.Rect(WIDTH // 2 - 75, HEIGHT - 60, 150, 40)
    pygame.draw.rect(screen, BUTTON_COLOR, rect, border_radius=10)
    pygame.draw.rect(screen, BUTTON_BORDER, rect, 2, border_radius=10)
    text = INFO_FONT.render("Restart", True, (0, 0, 0))
    screen.blit(text, (WIDTH // 2 - text.get_width() // 2, HEIGHT - 55))
    return rect


# Game loop
running = True
while running:
    screen.fill(BG_COLOR)
    draw_grid()
    draw_marks()
    if game_over:
        if win_combo:
            draw_win_line(win_combo)
        draw_result()
        reset_button = draw_reset_button()

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN and not game_over:
            x, y = event.pos
            if y < WIDTH:
                idx = (y // CELL_SIZE) * 3 + (x // CELL_SIZE)
                if board[idx] == ' ':
                    board[idx] = 'X'
                    combo = check_winner('X')
                    if combo:
                        game_over = True
                        winner = 'X'
                        win_combo = combo
                    elif is_draw():
                        game_over = True
                    else:
                        ai_move = best_ai_move()
                        board[ai_move] = 'O'
                        combo = check_winner('O')
                        if combo:
                            game_over = True
                            winner = 'O'
                            win_combo = combo
                        elif is_draw():
                            game_over = True
        elif event.type == pygame.MOUSEBUTTONDOWN and game_over:
            if reset_button.collidepoint(event.pos):
                reset()

    pygame.display.flip()
    clock.tick(60)

pygame.quit()
sys.exit()
