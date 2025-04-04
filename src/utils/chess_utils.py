import math
import warnings
from io import StringIO
from typing import Any, Callable, Iterable, List, Union

import chess
import chess.pgn
import chess.svg
import regex as re


def check_token_validity(predicted_uci: str, valid_uci: str):
    """This function checks whether the output of an LLM is semantically valid.
    This is a bit more technical than simply checking if the moves are valid, because
    tokenization typically splits the moves into sub-plies, and we need to check each
    independent of if the other parts are invalid."""

    board = chess.Board()
    valid_uci = valid_uci.lower()
    predicted_uci = predicted_uci.lower()

    grades = ""
    for plin, plout in zip(valid_uci.split(), predicted_uci.split()):
        from_sq, to_sq = plout[:2], plout[2:]
        start_found = False
        end_found = False
        for legal_move in board.generate_legal_moves():
            legal_from = legal_move.uci()[:2]

            if legal_from == from_sq:
                start_found = True
            if legal_from == plin[:2]:
                legal_to = legal_move.uci()[2:]
                if legal_to == to_sq:
                    end_found = True

        grades += ".." if start_found else "XX"
        grades += ".." if end_found else "X" * len(to_sq)
        grades += " "
        board.push_uci(plin)

    print(valid_uci, predicted_uci, grades, sep="\n")


def uci_to_board(
    uci_moves: Union[str, Iterable],
    *,
    force=False,
    fail_silent=False,
    verbose=True,
    as_board_stack=False,
    map_function: Callable = lambda x: x,
    reset_halfmove_clock=False,
) -> Union[chess.Board, List[Union[chess.Board, Any]]]:
    """Returns a chess.Board object from a string of UCI moves
    Params:
        force: If true, illegal moves are forcefully made. O/w, the error is thrown
        verbose: Alert user via prints that illegal moves were attempted."""
    board = chess.Board()
    forced_moves = []
    did_force = False
    board_stack = [map_function(board.copy())]

    if isinstance(uci_moves, str):
        uci_moves = uci_moves.lower().split(" ")

    for i, move in enumerate(uci_moves):
        try:
            move_obj = board.parse_uci(move)
            if reset_halfmove_clock:
                board.halfmove_clock = 0
            board.push(move_obj)
        except (chess.IllegalMoveError, chess.InvalidMoveError) as ex:
            if force:
                did_force = True
                forced_moves.append((i, move))
                piece = board.piece_at(chess.parse_square(move[:2]))
                board.set_piece_at(chess.parse_square(move[:2]), None)
                board.set_piece_at(chess.parse_square(move[2:4]), piece)
            elif fail_silent:
                if as_board_stack:
                    return board_stack
                else:
                    return map_function(board)
            else:
                if verbose:
                    print(f"Failed on (move_id, uci): ({i},{move})")
                    if as_board_stack:
                        return board_stack
                    else:
                        return map_function(board)
                else:
                    raise ex
        board_stack.append(map_function(board.copy()))
    if verbose and did_force:
        print(f"Forced (move_id, uci): {forced_moves}")

    if as_board_stack:
        return board_stack
    else:
        return map_function(board)


def pgn_to_uci(pgn_string: str):
    """
    Converts a pgn string into uci notation.
    Example usage:
    ```
    >>> pgn_to_uci('1.e4 e5 2.Nf3 Nc6 3.Bb5')
    'e2e4 e7e5 g1f3 b8c6 f1b5'
    ```
    """

    pgn_io = StringIO(pgn_string)
    game = chess.pgn.read_game(pgn_io)
    return pgn_game_to_uci_moves(game)


def pgn_game_to_uci_moves(pgn_game: chess.pgn.Game):
    return " ".join([m.uci() for m in pgn_game.mainline_moves()])


def uci_to_pgn(
    uci_string: str,
    headers=dict(
        Event="?",
        Site="?",
        Date="????.??.??",
        Round="?",
        White="?",
        Black="?",
        Result="*",
    ),
):
    """
    Converts a uci string into pgn.

    Example usage (**using print**):
    ```
    >>> print(uci_to_pgn('e2e4 e7e5 g1f3 b8c6 f1b5'))
    [Event "?"]
    [Site "?"]
    [Date "????.??.??"]
    [Round "?"]
    [White "?"]
    [Black "?"]
    [Result "*"]

    1. e4 e5 2. Nf3 Nc6 3. Bb5 *
    ```
    """

    game_pgn = chess.pgn.Game()
    game_pgn.headers.update(**headers)
    node = game_pgn
    for ply, move in enumerate(uci_string.lower().split()):
        try:
            node = node.add_variation(chess.Move.from_uci(move))
        except AssertionError as ex:
            warnings.warn(
                f"Warning! UCI cannot be converted to PGN on ply {ply}: '{move}'."
            )
            break
    return game_pgn


def win2cp(win_percent: float):
    """
    Convert a win percentage to centipawn evaluation.

    Parameters:
    win_percent (float): The win percentage, a value between 0 and 1.

    Returns:
    float: The centipawn evaluation. Returns positive infinity for a win percentage of 1,
           negative infinity for a win percentage of 0, and a calculated centipawn value otherwise.

    (Formula derived from https://lichess.org/page/accuracy)
    """
    if win_percent == 1.0:
        return math.inf
    if win_percent == 0.0:
        return -math.inf

    return (6250000 * math.log(-win_percent / (win_percent - 1))) / 23013


def cp2win(centipawns: float):
    """
    Convert centipawn evaluation to win percentage.

    Parameters:
    centipawns (float): The evaluation in centipawns.

    Returns:
    float: The win percentage corresponding to the centipawn evaluation.

    (Formula derived from https://lichess.org/page/accuracy)
    """

    return 1 - 1 / (math.exp(0.00368208 * centipawns) + 1)


def move_accuracy(win_percent_before: float, win_percent_after: float):
    """
    Computes the accuracy of a move as a function of the win percentage before and after the move.

    Parameters:
    win_percent_before (float): The win percentage before the move.
    win_percent_after (float): The win percentage after the move.

    Returns:
    float: The computed accuracy of the move.
    """

    delta = win_percent_before - win_percent_after
    return 103.1668 * math.exp(-0.04354 * delta) - 3.1669
