use std::str::FromStr;

use chess::{Board, BoardBuilder, Color, Piece, Square, File, Rank};

pub fn decode_board(pieces: Vec<u8>, black: bool) -> Option<String> {
    if pieces.len() != 64 {
        return None;
    }

    let mut builder = BoardBuilder::new();
    for (index, piece) in pieces.into_iter().enumerate() {
        if let Some((piece, color)) = decode_piece(piece) {
            let (rank, file) = {
                if black {
                    (Rank::from_index(index / 8), File::from_index(index % 8))
                } else {
                    (Rank::from_index(7 - index / 8), File::from_index(index % 8))
                }
            };

            builder.piece(Square::make_square(rank, file), piece, color);
        }
    }

    builder.side_to_move(if black {
        Color::Black
    } else {
        Color::White
    });

    TryInto::<Board>::try_into(builder)
        .ok()
        .map(|board| board.to_string())
}

pub fn decode_piece(piece: u8) -> Option<(Piece, Color)> {
    if piece == 0 || piece > 12 {
        return None;
    }

    Some(match piece {
        1 => (Piece::Pawn, Color::White),
        2 => (Piece::Pawn, Color::Black),
        3 => (Piece::Bishop, Color::White),
        4 => (Piece::Bishop, Color::Black),
        5 => (Piece::Knight, Color::White),
        6 => (Piece::Knight, Color::Black),
        7 => (Piece::Rook, Color::White),
        8 => (Piece::Rook, Color::Black),
        9 => (Piece::Queen, Color::White),
        10 => (Piece::Queen, Color::Black),
        11 => (Piece::King, Color::White),
        12 => (Piece::King, Color::Black),
        _ => unreachable!(),
    })
}

pub fn decode_move(mov: &str, black: bool) -> Option<(u8, u8)> {
    let (from_str, to_str) = mov.split_at(2);

    let from = decode_square(from_str, black)?;
    let to = decode_square(to_str, black)?;

    Some((from, to))
}

pub fn decode_square(sq: &str, black: bool) -> Option<u8> {
    let square = Square::from_str(sq).ok()?;

    if black {
        Some((8 * square.get_rank().to_index() + square.get_file().to_index()) as u8)
    } else {
        Some((8 * (7 - square.get_rank().to_index()) + square.get_file().to_index()) as u8)
    }
}
