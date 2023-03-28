import white_pawn from './svg/white_pawn.svg';
import black_pawn from './svg/black_pawn.svg';

import white_bishop from './svg/white_bishop.svg';
import black_bishop from './svg/black_bishop.svg';

import white_knight from './svg/white_knight.svg';
import black_knight from './svg/black_knight.svg';

import white_rook from './svg/white_rook.svg';
import black_rook from './svg/black_rook.svg';

import white_queen from './svg/white_queen.svg';
import black_queen from './svg/black_queen.svg';

import white_king from './svg/white_king.svg';
import black_king from './svg/black_king.svg';

export const pieces = new Map();

pieces.set(1, white_pawn);
pieces.set(2, black_pawn);

pieces.set(3, white_bishop);
pieces.set(4, black_bishop);

pieces.set(5, white_knight);
pieces.set(6, black_knight);

pieces.set(7, white_rook);
pieces.set(8, black_rook);

pieces.set(9, white_queen);
pieces.set(10, black_queen);

pieces.set(11, white_king);
pieces.set(12, black_king);

export const startPosition = [
  8, 6, 4, 10, 12, 4, 6, 8,
  2, 2, 2, 2, 2, 2, 2, 2,
  0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0,
  1, 1, 1, 1, 1, 1, 1, 1,
  7, 5, 3, 9, 11, 3, 5, 7
];

export const sounds = () => ({
  thock1: new Audio('/sounds/thock1.mp3'),
  thock2: new Audio('/sounds/thock2.mp3'),
  thack: new Audio('/sounds/thack.mp3'),
});
