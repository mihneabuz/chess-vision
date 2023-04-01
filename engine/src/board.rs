use chess::{Board, BoardBuilder, Color, Piece, Square};

use crate::Request;

pub fn decode(payload: Request) -> Option<String> {
    if payload.pieces.len() != 64 {
        return None;
    }

    let mut builder = BoardBuilder::new();
    for (index, piece) in payload.pieces.into_iter().enumerate() {
        if let Some((piece, color)) = decode_piece(piece) {
            let square = unsafe {
                if payload.black {
                    Square::new(63 - index as u8)
                } else {
                    Square::new(index as u8)
                }
            };

            builder.piece(square, piece, color);
        }
    }

    builder.side_to_move(if payload.black {
        Color::Black
    } else {
        Color::White
    });

    TryInto::<Board>::try_into(builder)
        .ok()
        .map(|board| board.to_string())
}

fn decode_piece(piece: u8) -> Option<(Piece, Color)> {
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
