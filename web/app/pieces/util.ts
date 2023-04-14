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

export const pieces = new Map([
  [1, white_pawn],
  [2, black_pawn],
  [3, white_bishop],
  [4, black_bishop],
  [5, white_knight],
  [6, black_knight],
  [7, white_rook],
  [8, black_rook],
  [9, white_queen],
  [10, black_queen],
  [11, white_king],
  [12, black_king],
]);

export const squares = [
  'a8', 'b8', 'c8', 'd8', 'e8', 'f8', 'g8', 'h8',
  'a7', 'b7', 'c7', 'd7', 'e7', 'f7', 'g7', 'h7',
  'a6', 'b6', 'c6', 'd6', 'e6', 'f6', 'g6', 'h6',
  'a5', 'b5', 'c5', 'd5', 'e5', 'f5', 'g5', 'h5',
  'a4', 'b4', 'c4', 'd4', 'e4', 'f4', 'g4', 'h4',
  'a3', 'b3', 'c3', 'd3', 'e3', 'f3', 'g3', 'h3',
  'a2', 'b2', 'c2', 'd2', 'e2', 'f2', 'g2', 'h2',
  'a1', 'b1', 'c1', 'd1', 'e1', 'f1', 'g1', 'h1',
]

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

export const squareToPos = new Map(
  squares.map((square, index) => [square, index])
);

export const sounds = () => ({
  thock1: new Audio('/sounds/thock1.mp3'),
  thock2: new Audio('/sounds/thock2.mp3'),
  thack: new Audio('/sounds/thack.mp3'),
});
