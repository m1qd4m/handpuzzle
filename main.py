"""
main.py
-------
Hand-controlled 3x3 photo puzzle game.

Game states
-----------
  START_SCREEN  – split layout: webcam feed left, instructions right
                  Make a FIST to begin countdown (SPACE also works)
  COUNTDOWN     – 3-second visual countdown; photo captured at end
  GAMEPLAY      – puzzle interaction via hand tracking
  SOLVED        – victory overlay; fist or R to restart
"""

import time
import cv2
import numpy as np

from hand_tracker import HandTracker
from puzzle_manager import PuzzleManager

# ─────────────────────────────────────────────────────────────
# Layout constants
# ─────────────────────────────────────────────────────────────
CANVAS_W   = 1100
CANVAS_H   = 700
BOARD_SIZE = 600

# Puzzle board: vertically centred, on the RIGHT side
BOARD_X = CANVAS_W - BOARD_SIZE - 20
BOARD_Y = (CANVAS_H - BOARD_SIZE) // 2

# Webcam panel: LEFT side, same height as board
CAM_W = CANVAS_W - BOARD_SIZE - 60   # ~480px
CAM_H = BOARD_SIZE
CAM_X = 20
CAM_Y = BOARD_Y

# Instruction panel sits inside the left column below cam
# (only used on START_SCREEN where cam is smaller)
START_CAM_W = CAM_W
START_CAM_H = int(CAM_W * 480 / 640)   # keep 4:3 aspect
START_CAM_X = CAM_X
START_CAM_Y = CAM_Y

COUNTDOWN_SECS = 3

# ─────────────────────────────────────────────────────────────
# Colours (BGR)
# ─────────────────────────────────────────────────────────────
CLR_BG        = (18,  18,  28)
CLR_PANEL     = (30,  30,  45)
CLR_OPEN      = (50, 220,  80)    # green  – open hand / cursor
CLR_GRAB      = (50,  80, 230)    # red    – pinching
CLR_FIST      = (0,  200, 255)    # yellow – fist detected
CLR_WHITE     = (255, 255, 255)
CLR_GREY      = (160, 160, 160)
CLR_ACCENT    = (80,  220,  80)   # title green
CLR_GRID      = (180, 180, 180)
CLR_WIN       = (0,   200,   0)
CLR_BORDER    = (60,   60,  90)

CURSOR_RADIUS = 16
FONT          = cv2.FONT_HERSHEY_DUPLEX
FONT_BOLD     = cv2.FONT_HERSHEY_SIMPLEX


# ─────────────────────────────────────────────────────────────
# Text helpers
# ─────────────────────────────────────────────────────────────
def text_size(text, scale, thickness, font=FONT):
    (tw, th), bl = cv2.getTextSize(text, font, scale, thickness)
    return tw, th + bl


def draw_text_centred(canvas, text, cx, cy, scale=1.0,
                      colour=CLR_WHITE, thickness=2, font=FONT):
    tw, th = text_size(text, scale, thickness, font)
    cv2.putText(canvas, text,
                (cx - tw // 2, cy + th // 2),
                font, scale, colour, thickness, cv2.LINE_AA)


def draw_text_shadow(canvas, text, cx, cy, scale=1.0,
                     colour=CLR_WHITE, thickness=2, font=FONT):
    draw_text_centred(canvas, text, cx + 2, cy + 2, scale,
                      (0, 0, 0), thickness + 1, font)
    draw_text_centred(canvas, text, cx, cy, scale, colour, thickness, font)


def draw_text_left(canvas, text, x, y, scale=0.65,
                   colour=CLR_WHITE, thickness=1, font=FONT):
    cv2.putText(canvas, text, (x, y), font, scale,
                colour, thickness, cv2.LINE_AA)


# ─────────────────────────────────────────────────────────────
# State
# ─────────────────────────────────────────────────────────────
class State:
    START_SCREEN = "START_SCREEN"
    COUNTDOWN    = "COUNTDOWN"
    GAMEPLAY     = "GAMEPLAY"
    SOLVED       = "SOLVED"


# ─────────────────────────────────────────────────────────────
# Fist cooldown helper — prevents instant re-trigger
# ─────────────────────────────────────────────────────────────
FIST_COOLDOWN = 1.5   # seconds


# ─────────────────────────────────────────────────────────────
# Main app
# ─────────────────────────────────────────────────────────────
class PuzzleApp:

    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise RuntimeError("Could not open webcam.")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        self.tracker = HandTracker()
        self.puzzle: PuzzleManager | None = None

        self.state              = State.START_SCREEN
        self._countdown_start   = 0.0
        self._fist_trigger_time = 0.0   # last time fist was used to change state

    # ─────────────────────────────────────────
    # Main loop
    # ─────────────────────────────────────────
    def run(self):
        cv2.namedWindow("Hand Puzzle", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Hand Puzzle", CANVAS_W, CANVAS_H)

        while True:
            canvas = self._blank_canvas()

            ret, raw = self.cap.read()
            if not ret:
                continue
            raw = cv2.flip(raw, 1)

            # Always process hand so fist/pinch states are fresh
            rgb = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
            cursor, is_pinching, is_fist = self.tracker.process(rgb)

            if self.state == State.START_SCREEN:
                self._handle_start_screen(canvas, raw, cursor, is_fist)

            elif self.state == State.COUNTDOWN:
                self._handle_countdown(canvas, raw)

            elif self.state == State.GAMEPLAY:
                self._handle_gameplay(canvas, raw, cursor, is_pinching)

            elif self.state == State.SOLVED:
                self._handle_solved(canvas, raw, is_fist)

            cv2.imshow("Hand Puzzle", canvas)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            self._handle_key(key)

        self._cleanup()

    # ─────────────────────────────────────────
    # START SCREEN  —  split layout
    # ─────────────────────────────────────────
    def _handle_start_screen(self, canvas, raw, cursor, is_fist):
        # ── Left panel background ──
        cv2.rectangle(canvas,
                      (CAM_X - 10, CAM_Y - 10),
                      (CAM_X + CAM_W + 10, CAM_Y + CAM_H + 10),
                      CLR_PANEL, -1)

        # ── Webcam feed (top of left panel) ──
        cam = cv2.resize(raw, (START_CAM_W, START_CAM_H))
        canvas[START_CAM_Y:START_CAM_Y + START_CAM_H,
               START_CAM_X:START_CAM_X + START_CAM_W] = cam

        # ── Fist indicator on webcam ──
        border_clr = CLR_FIST if is_fist else CLR_OPEN
        cv2.rectangle(canvas,
                      (START_CAM_X, START_CAM_Y),
                      (START_CAM_X + START_CAM_W, START_CAM_Y + START_CAM_H),
                      border_clr, 4)

        # Live status label
        status_text = "FIST DETECTED!" if is_fist else "Show your hand..."
        status_clr  = CLR_FIST        if is_fist else CLR_GREY
        draw_text_shadow(canvas, status_text,
                         START_CAM_X + START_CAM_W // 2,
                         START_CAM_Y + START_CAM_H + 22,
                         scale=0.7, colour=status_clr, thickness=1)

        # ── Instruction panel (below webcam) ──
        inst_y = START_CAM_Y + START_CAM_H + 55
        inst_x = CAM_X + 10

        sections = [
            # (label, lines, label_colour)
            ("HOW TO START",
             ["Make a FIST  (keep it 1 sec)",
              "OR press  SPACE"],
             (80, 220, 80)),
            ("HOW TO PLAY",
             ["Pinch fingers = GRAB tile",
              "Open hand     = RELEASE tile",
              "Swap all tiles to solve!"],
             (80, 180, 255)),
            ("CONTROLS",
             ["Q  =  Quit",
              "R  =  Restart (on win screen)"],
             (200, 200, 100)),
        ]

        for label, lines, lclr in sections:
            # Section heading
            cv2.rectangle(canvas,
                          (inst_x - 5, inst_y - 16),
                          (inst_x + CAM_W - 15, inst_y + 4),
                          CLR_PANEL, -1)
            draw_text_left(canvas, label, inst_x, inst_y,
                           scale=0.55, colour=lclr, thickness=1)
            inst_y += 22
            for line in lines:
                draw_text_left(canvas, line, inst_x + 8, inst_y,
                               scale=0.58, colour=CLR_WHITE, thickness=1)
                inst_y += 20
            inst_y += 10   # gap between sections

        # ── Right panel — title + gesture guide ──
        right_cx = BOARD_X + BOARD_SIZE // 2

        # Title
        draw_text_shadow(canvas, "HAND",
                         right_cx, BOARD_Y + 80,
                         scale=3.5, colour=CLR_ACCENT, thickness=4)
        draw_text_shadow(canvas, "PUZZLE",
                         right_cx, BOARD_Y + 160,
                         scale=3.5, colour=CLR_ACCENT, thickness=4)

        # Divider
        cv2.line(canvas,
                 (BOARD_X + 40, BOARD_Y + 195),
                 (BOARD_X + BOARD_SIZE - 40, BOARD_Y + 195),
                 CLR_BORDER, 2)

        # Gesture cards
        gestures = [
            ("OPEN HAND",  "Move cursor",   CLR_OPEN,  "o"),
            ("PINCH",      "Grab a tile",   CLR_GRAB,  "p"),
            ("FIST",       "Start game",    CLR_FIST,  "f"),
        ]
        card_y = BOARD_Y + 230
        card_h = 90
        card_gap = 15
        card_x = BOARD_X + 30
        card_w = BOARD_SIZE - 60

        for i, (gname, gdesc, gclr, _) in enumerate(gestures):
            cy = card_y + i * (card_h + card_gap)
            # Card background
            cv2.rectangle(canvas,
                          (card_x, cy),
                          (card_x + card_w, cy + card_h),
                          CLR_PANEL, -1)
            cv2.rectangle(canvas,
                          (card_x, cy),
                          (card_x + card_w, cy + card_h),
                          gclr, 2)
            # Colour dot
            cv2.circle(canvas, (card_x + 40, cy + card_h // 2), 18, gclr, -1)
            cv2.circle(canvas, (card_x + 40, cy + card_h // 2), 20, CLR_WHITE, 1)
            # Text
            draw_text_left(canvas, gname,
                           card_x + 72, cy + 30,
                           scale=0.75, colour=CLR_WHITE, thickness=2)
            draw_text_left(canvas, gdesc,
                           card_x + 72, cy + 60,
                           scale=0.65, colour=CLR_GREY, thickness=1)

        # Bottom hint
        draw_text_shadow(canvas, "Position hand in camera view, then make a FIST",
                         right_cx, BOARD_Y + BOARD_SIZE - 20,
                         scale=0.55, colour=CLR_GREY, thickness=1)

        # ── Fist trigger ──
        now = time.time()
        if is_fist and (now - self._fist_trigger_time) > FIST_COOLDOWN:
            self._fist_trigger_time = now
            self._countdown_start   = now
            self.state = State.COUNTDOWN

    # ─────────────────────────────────────────
    # COUNTDOWN
    # ─────────────────────────────────────────
    def _handle_countdown(self, canvas, raw):
        elapsed   = time.time() - self._countdown_start
        remaining = COUNTDOWN_SECS - elapsed

        # Webcam on left
        cam = cv2.resize(raw, (CAM_W, CAM_H))
        canvas[CAM_Y:CAM_Y + CAM_H, CAM_X:CAM_X + CAM_W] = cam
        cv2.rectangle(canvas, (CAM_X, CAM_Y),
                      (CAM_X + CAM_W, CAM_Y + CAM_H), CLR_ACCENT, 3)

        # Right: countdown
        right_cx = BOARD_X + BOARD_SIZE // 2
        right_cy = BOARD_Y + BOARD_SIZE // 2

        if remaining > 0:
            digit = str(int(remaining) + 1)
            draw_text_shadow(canvas, "Get ready!", right_cx, right_cy - 90,
                             scale=1.2, colour=CLR_WHITE)
            draw_text_shadow(canvas, digit, right_cx, right_cy,
                             scale=8.0, colour=CLR_ACCENT, thickness=6)
            draw_text_shadow(canvas, "Smile for the puzzle :)",
                             right_cx, right_cy + 120,
                             scale=0.75, colour=CLR_GREY, thickness=1)

            # Progress bar
            bar_w = BOARD_SIZE - 80
            bar_x = BOARD_X + 40
            bar_y = BOARD_Y + BOARD_SIZE - 50
            progress = 1.0 - (remaining / COUNTDOWN_SECS)
            cv2.rectangle(canvas, (bar_x, bar_y),
                          (bar_x + bar_w, bar_y + 14), CLR_PANEL, -1)
            cv2.rectangle(canvas, (bar_x, bar_y),
                          (bar_x + int(bar_w * progress), bar_y + 14),
                          CLR_ACCENT, -1)
            cv2.rectangle(canvas, (bar_x, bar_y),
                          (bar_x + bar_w, bar_y + 14), CLR_BORDER, 2)
        else:
            photo = cv2.resize(raw, (BOARD_SIZE, BOARD_SIZE))
            self.puzzle = PuzzleManager(photo, board_origin=(BOARD_X, BOARD_Y))
            self.state  = State.GAMEPLAY

    # ─────────────────────────────────────────
    # GAMEPLAY
    # ─────────────────────────────────────────
    def _handle_gameplay(self, canvas, raw, cursor, is_pinching):
        assert self.puzzle is not None

        scaled_cursor = self._scale_cursor(cursor, raw.shape)
        self.puzzle.update(scaled_cursor, is_pinching)

        # ── Left: mini webcam ──
        mini_h = CAM_H // 2
        mini_w = CAM_W
        mini   = cv2.resize(raw, (mini_w, mini_h))
        my     = CAM_Y + (CAM_H - mini_h) // 2
        canvas[my:my + mini_h, CAM_X:CAM_X + mini_w] = mini
        cv2.rectangle(canvas, (CAM_X, my),
                      (CAM_X + mini_w, my + mini_h), CLR_BORDER, 2)

        # ── Left: live HUD below mini cam ──
        hud_y = my + mini_h + 20
        hud_x = CAM_X + 10

        state_text = "GRABBING" if is_pinching else "OPEN"
        state_clr  = CLR_GRAB  if is_pinching else CLR_OPEN
        draw_text_left(canvas, f"Hand: {state_text}",
                       hud_x, hud_y, scale=0.7, colour=state_clr, thickness=2)

        tips = [
            "PINCH  =  grab tile",
            "RELEASE  =  drop tile",
            "Swap tiles to restore",
            "the original photo!",
        ]
        for i, tip in enumerate(tips):
            draw_text_left(canvas, tip,
                           hud_x, hud_y + 35 + i * 24,
                           scale=0.6, colour=CLR_GREY, thickness=1)

        # ── Right: board ──
        self._draw_board(canvas)

        # Title above board
        draw_text_shadow(canvas, "HAND PUZZLE",
                         BOARD_X + BOARD_SIZE // 2, BOARD_Y - 28,
                         scale=1.0, colour=CLR_ACCENT, thickness=2)

        # ── Cursor ──
        if scaled_cursor is not None:
            clr = CLR_GRAB if is_pinching else CLR_OPEN
            cx, cy = scaled_cursor
            cv2.circle(canvas, (cx, cy), CURSOR_RADIUS, clr, -1)
            cv2.circle(canvas, (cx, cy), CURSOR_RADIUS + 2, CLR_WHITE, 2)

        if self.puzzle.is_solved():
            self.state = State.SOLVED

    # ─────────────────────────────────────────
    # SOLVED
    # ─────────────────────────────────────────
    def _handle_solved(self, canvas, raw, is_fist):
        assert self.puzzle is not None

        self._draw_board(canvas)

        # Semi-transparent overlay on board
        overlay = canvas.copy()
        cv2.rectangle(overlay,
                      (BOARD_X, BOARD_Y),
                      (BOARD_X + BOARD_SIZE, BOARD_Y + BOARD_SIZE),
                      (0, 160, 0), -1)
        cv2.addWeighted(overlay, 0.5, canvas, 0.5, 0, canvas)

        cx = BOARD_X + BOARD_SIZE // 2
        draw_text_shadow(canvas, "PUZZLE",   cx, BOARD_Y + 220,
                         scale=3.0, colour=CLR_WHITE, thickness=4)
        draw_text_shadow(canvas, "SOLVED!",  cx, BOARD_Y + 310,
                         scale=3.0, colour=CLR_ACCENT, thickness=4)
        draw_text_shadow(canvas, "Make a FIST  or  press R  to play again",
                         cx, BOARD_Y + 420,
                         scale=0.7, colour=CLR_WHITE, thickness=1)

        # Left: webcam still running
        cam = cv2.resize(raw, (CAM_W, CAM_H // 2))
        cy0 = CAM_Y + CAM_H // 4
        canvas[cy0:cy0 + CAM_H // 2, CAM_X:CAM_X + CAM_W] = cam

        # Fist to restart
        now = time.time()
        if is_fist and (now - self._fist_trigger_time) > FIST_COOLDOWN:
            self._fist_trigger_time = now
            self.puzzle = None
            self.state  = State.START_SCREEN

    # ─────────────────────────────────────────
    # Rendering helpers
    # ─────────────────────────────────────────
    def _blank_canvas(self):
        return np.full((CANVAS_H, CANVAS_W, 3), CLR_BG, dtype=np.uint8)

    def _draw_board(self, canvas):
        assert self.puzzle is not None
        ps   = self.puzzle.piece_size
        data = self.puzzle.render_data()

        for item in sorted(data, key=lambda d: d["is_grabbed"]):
            x, y = item["x"], item["y"]
            tile  = item["image"]

            x1, y1 = max(x, 0), max(y, 0)
            x2, y2 = min(x + ps, CANVAS_W), min(y + ps, CANVAS_H)
            sx1 = x1 - x
            sy1 = y1 - y
            sx2 = sx1 + (x2 - x1)
            sy2 = sy1 + (y2 - y1)

            if x2 <= x1 or y2 <= y1:
                continue

            if item["is_grabbed"]:
                off = 6
                shadow_x1 = max(x1 + off, 0)
                shadow_y1 = max(y1 + off, 0)
                shadow_x2 = min(x2 + off, CANVAS_W)
                shadow_y2 = min(y2 + off, CANVAS_H)
                canvas[shadow_y1:shadow_y2, shadow_x1:shadow_x2] = (
                    canvas[shadow_y1:shadow_y2, shadow_x1:shadow_x2] * 0.4
                ).astype(np.uint8)

            canvas[y1:y2, x1:x2] = tile[sy1:sy2, sx1:sx2]

            if item["is_grabbed"]:
                cv2.rectangle(canvas, (x1, y1), (x2 - 1, y2 - 1),
                              (0, 220, 255), 3)

        # Grid lines
        for i in range(1, 3):
            lx = BOARD_X + i * ps
            cv2.line(canvas, (lx, BOARD_Y), (lx, BOARD_Y + BOARD_SIZE), CLR_GRID, 2)
            ly = BOARD_Y + i * ps
            cv2.line(canvas, (BOARD_X, ly), (BOARD_X + BOARD_SIZE, ly), CLR_GRID, 2)

        cv2.rectangle(canvas,
                      (BOARD_X, BOARD_Y),
                      (BOARD_X + BOARD_SIZE, BOARD_Y + BOARD_SIZE),
                      CLR_WHITE, 3)

    def _scale_cursor(self, cursor, cam_shape):
        if cursor is None:
            return None
        cam_h, cam_w = cam_shape[:2]
        cx, cy = cursor
        sx = int(cx / cam_w * BOARD_SIZE) + BOARD_X
        sy = int(cy / cam_h * BOARD_SIZE) + BOARD_Y
        return sx, sy

    # ─────────────────────────────────────────
    # Key handling
    # ─────────────────────────────────────────
    def _handle_key(self, key):
        if self.state == State.START_SCREEN and key == ord(' '):
            self._countdown_start = time.time()
            self.state = State.COUNTDOWN

        elif self.state == State.SOLVED and key == ord('r'):
            self.puzzle = None
            self.state  = State.START_SCREEN

    # ─────────────────────────────────────────
    # Cleanup
    # ─────────────────────────────────────────
    def _cleanup(self):
        self.tracker.close()
        self.cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    app = PuzzleApp()
    app.run()
