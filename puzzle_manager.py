"""
puzzle_manager.py
-----------------
3×3 sliding photo puzzle logic:
  - PuzzlePiece  : data class for each tile
  - PuzzleManager: slicing, shuffling, drag-and-drop, win detection
"""

import random
import numpy as np


GRID_SIZE = 3          # 3×3 grid
NUM_PIECES = GRID_SIZE * GRID_SIZE   # 9


class PuzzlePiece:
    """Stores state for one puzzle tile."""

    def __init__(self, original_index: int, image_slice: np.ndarray):
        self.original_index: int = original_index   # solved position
        self.current_index: int = original_index    # current grid slot
        self.image: np.ndarray = image_slice        # tile pixels

        # Drag state
        self.is_grabbed: bool = False
        self.drag_x: int = 0   # centre of piece while being dragged
        self.drag_y: int = 0


class PuzzleManager:
    """Manages the 3×3 puzzle grid."""

    def __init__(self, source_image: np.ndarray, board_origin: tuple[int, int] = (0, 0)):
        """
        Parameters
        ----------
        source_image  : square BGR image (e.g. 600×600)
        board_origin  : (x, y) pixel offset of the board's top-left corner
                        on the display canvas.
        """
        h, w = source_image.shape[:2]
        assert h == w, "source_image must be square"

        self.board_size: int = h                  # e.g. 600
        self.piece_size: int = h // GRID_SIZE      # e.g. 200
        self.origin: tuple[int, int] = board_origin

        self.pieces: list[PuzzlePiece] = []
        self._grabbed_piece: PuzzlePiece | None = None

        self._slice_and_shuffle(source_image)

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _slice_and_shuffle(self, img: np.ndarray):
        ps = self.piece_size
        pieces = []
        for idx in range(NUM_PIECES):
            r, c = divmod(idx, GRID_SIZE)
            tile = img[r * ps:(r + 1) * ps, c * ps:(c + 1) * ps].copy()
            pieces.append(PuzzlePiece(original_index=idx, image_slice=tile))

        # Shuffle until not already solved
        indices = list(range(NUM_PIECES))
        while True:
            random.shuffle(indices)
            if indices != list(range(NUM_PIECES)):
                break

        for piece, new_idx in zip(pieces, indices):
            piece.current_index = new_idx

        self.pieces = pieces

    # ------------------------------------------------------------------
    # Coordinate helpers
    # ------------------------------------------------------------------

    def grid_slot_center(self, grid_index: int) -> tuple[int, int]:
        """Pixel centre of a grid slot (on the display canvas)."""
        r, c = divmod(grid_index, GRID_SIZE)
        ox, oy = self.origin
        ps = self.piece_size
        cx = ox + c * ps + ps // 2
        cy = oy + r * ps + ps // 2
        return cx, cy

    def grid_slot_topleft(self, grid_index: int) -> tuple[int, int]:
        r, c = divmod(grid_index, GRID_SIZE)
        ox, oy = self.origin
        ps = self.piece_size
        return ox + c * ps, oy + r * ps

    def cursor_to_grid_index(self, cx: int, cy: int) -> int | None:
        """Return grid index under cursor, or None if outside board."""
        ox, oy = self.origin
        lx = cx - ox
        ly = cy - oy
        ps = self.piece_size
        if 0 <= lx < self.board_size and 0 <= ly < self.board_size:
            return (ly // ps) * GRID_SIZE + (lx // ps)
        return None

    # ------------------------------------------------------------------
    # Interaction
    # ------------------------------------------------------------------

    def update(self, cursor: tuple[int, int] | None, is_pinching: bool):
        """
        Call once per frame with the current cursor position and pinch state.
        """
        if cursor is None:
            # Lost tracking — release any grabbed piece in place
            if self._grabbed_piece is not None:
                self._drop_piece(self._grabbed_piece, None)
            return

        cx, cy = cursor

        if is_pinching:
            if self._grabbed_piece is None:
                # Try to grab a piece under the cursor
                slot = self.cursor_to_grid_index(cx, cy)
                if slot is not None:
                    piece = self._piece_at_slot(slot)
                    if piece is not None:
                        piece.is_grabbed = True
                        piece.drag_x, piece.drag_y = cx, cy
                        self._grabbed_piece = piece
            else:
                # Continue dragging
                self._grabbed_piece.drag_x = cx
                self._grabbed_piece.drag_y = cy
        else:
            if self._grabbed_piece is not None:
                self._drop_piece(self._grabbed_piece, (cx, cy))
                self._grabbed_piece = None

    def _drop_piece(self, piece: PuzzlePiece, cursor: tuple[int, int] | None):
        """Find nearest slot and swap if occupied."""
        piece.is_grabbed = False

        if cursor is None:
            return  # Leave piece in current slot (no movement)

        # Find nearest grid slot to drop position
        best_slot = self._nearest_slot(*cursor)

        if best_slot == piece.current_index:
            return  # Dropped back on own slot — no change

        # Is there another piece in the target slot?
        other = self._piece_at_slot(best_slot)
        if other is not None:
            other.current_index = piece.current_index   # swap
        piece.current_index = best_slot

    def _nearest_slot(self, cx: int, cy: int) -> int:
        """Return the grid index whose centre is closest to (cx, cy)."""
        best_slot = 0
        best_dist = float("inf")
        for slot in range(NUM_PIECES):
            scx, scy = self.grid_slot_center(slot)
            dist = (scx - cx) ** 2 + (scy - cy) ** 2
            if dist < best_dist:
                best_dist = dist
                best_slot = slot
        return best_slot

    def _piece_at_slot(self, slot: int) -> "PuzzlePiece | None":
        for p in self.pieces:
            if p.current_index == slot:
                return p
        return None

    # ------------------------------------------------------------------
    # Win condition
    # ------------------------------------------------------------------

    def is_solved(self) -> bool:
        return all(p.current_index == p.original_index for p in self.pieces)

    # ------------------------------------------------------------------
    # Rendering data
    # ------------------------------------------------------------------

    def render_data(self) -> list[dict]:
        """
        Return a list of dicts describing where to draw each piece.
        Keys: 'image', 'x', 'y', 'is_grabbed'
          x, y = top-left pixel of piece on the canvas
        """
        ps = self.piece_size
        result = []
        for piece in self.pieces:
            if piece.is_grabbed:
                # Centre on cursor
                x = piece.drag_x - ps // 2
                y = piece.drag_y - ps // 2
            else:
                x, y = self.grid_slot_topleft(piece.current_index)
            result.append({
                "image": piece.image,
                "x": x,
                "y": y,
                "is_grabbed": piece.is_grabbed,
            })
        return result
