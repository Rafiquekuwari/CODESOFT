const boardEl = document.getElementById("board");
const infoEl = document.getElementById("info");
const restartBtn = document.getElementById("restart");

let board = [' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '];
let gameOver = false;
let playerTurn = true;
let winner = null;
let winCombo = null;

const winCombos = [
  [0, 1, 2], [3, 4, 5], [6, 7, 8],
  [0, 3, 6], [1, 4, 7], [2, 5, 8],
  [0, 4, 8], [2, 4, 6]
];

function drawBoard() {
  boardEl.innerHTML = '';
  board.forEach((cell, index) => {
    const cellEl = document.createElement('div');
    cellEl.classList.add('cell');
    cellEl.textContent = cell;
    cellEl.addEventListener('click', () => handleMove(index));
    boardEl.appendChild(cellEl);
  });
}

function handleMove(index) {
  if (gameOver || board[index] !== ' ') return;

  board[index] = 'X';
  if (checkWinner('X')) {
    gameOver = true;
    winner = 'X';
    winCombo = checkWinner('X');
  } else if (isDraw()) {
    gameOver = true;
  } else {
    setTimeout(() => aiMove(), 300);
  }
  drawBoard();
  updateInfo();
}

function aiMove() {
  const move = bestAiMove();
  board[move] = 'O';
  if (checkWinner('O')) {
    gameOver = true;
    winner = 'O';
    winCombo = checkWinner('O');
  } else if (isDraw()) {
    gameOver = true;
  }
  drawBoard();
  updateInfo();
}

function checkWinner(symbol) {
  for (const combo of winCombos) {
    if (board[combo[0]] === symbol && board[combo[1]] === symbol && board[combo[2]] === symbol) {
      return combo;
    }
  }
  return null;
}

function isDraw() {
  return !board.includes(' ');
}

function bestAiMove() {
  let bestScore = -Infinity;
  let bestMove;
  for (let i = 0; i < board.length; i++) {
    if (board[i] === ' ') {
      board[i] = 'O';
      const score = minimax(board, false);
      board[i] = ' ';
      if (score > bestScore) {
        bestScore = score;
        bestMove = i;
      }
    }
  }
  return bestMove;
}

function minimax(board, isMax) {
  const winner = checkWinner('O') ? 1 : checkWinner('X') ? -1 : 0;
  if (winner !== 0) return winner;

  if (isDraw()) return 0;

  let bestScore = isMax ? -Infinity : Infinity;
  for (let i = 0; i < board.length; i++) {
    if (board[i] === ' ') {
      board[i] = isMax ? 'O' : 'X';
      const score = minimax(board, !isMax);
      board[i] = ' ';
      bestScore = isMax ? Math.max(bestScore, score) : Math.min(bestScore, score);
    }
  }
  return bestScore;
}

function updateInfo() {
  if (gameOver) {
    if (winner) {
      infoEl.textContent = winner === 'X' ? 'You Win!' : 'AI Wins!';
    } else {
      infoEl.textContent = 'It\'s a Draw!';
    }
  } else {
    infoEl.textContent = playerTurn ? 'Your turn!' : 'AI\'s turn!';
  }
}

function resetGame() {
  board = [' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '];
  gameOver = false;
  winner = null;
  winCombo = null;
  drawBoard();
  updateInfo();
}

restartBtn.addEventListener('click', resetGame);

drawBoard();
updateInfo();
