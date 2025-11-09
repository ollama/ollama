#!/usr/bin/env node

/**
 * Terminal Snake Game
 * 
 * A retro-style snake game that runs in the terminal
 * 
 * Controls:
 *   w/a/s/d or arrow keys - Move
 *   q - Quit
 *   r - Restart
 *   p - Pause
 * 
 * Usage:
 *   node snake-game-standalone.js
 */

const readline = require('readline');

// Game configuration
const WIDTH = 30;
const HEIGHT = 20;
const INITIAL_SPEED = 200; // milliseconds

// Game state
let snake = [{ x: 15, y: 10 }];
let food = { x: 5, y: 5 };
let direction = { x: 1, y: 0 };
let score = 0;
let gameRunning = true;
let gamePaused = false;
let gameSpeed = INITIAL_SPEED;
let gameInterval;

// Initialize readline for input
const rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout
});

// Set terminal to raw mode for immediate key input
process.stdin.setRawMode(true);
process.stdin.resume();
process.stdin.setEncoding('utf8');

// Hide cursor
process.stdout.write('\x1b[?25l');

// Clear screen
function clearScreen() {
    process.stdout.write('\x1b[2J\x1b[H');
}

// Draw game board
function drawBoard() {
    clearScreen();
    
    // Header
    console.log('╔' + '═'.repeat(WIDTH) + '╗');
    console.log('║' + 'SNAKE GAME'.padStart((WIDTH + 10) / 2).padEnd(WIDTH) + '║');
    console.log('║' + `Score: ${score}`.padEnd(WIDTH) + '║');
    console.log('╠' + '═'.repeat(WIDTH) + '╣');
    
    // Game area
    for (let y = 0; y < HEIGHT; y++) {
        let line = '║';
        for (let x = 0; x < WIDTH; x++) {
            if (snake.some(seg => seg.x === x && seg.y === y)) {
                // Snake body
                if (snake[0].x === x && snake[0].y === y) {
                    line += '●'; // Head
                } else {
                    line += '○'; // Body
                }
            } else if (food.x === x && food.y === y) {
                line += '🍎'; // Food
            } else {
                line += ' '; // Empty
            }
        }
        line += '║';
        console.log(line);
    }
    
    // Footer
    console.log('╚' + '═'.repeat(WIDTH) + '╝');
    console.log('Controls: WASD/Arrows = Move | P = Pause | R = Restart | Q = Quit');
    if (gamePaused) {
        console.log('\n⏸ PAUSED - Press P to resume');
    }
}

// Generate new food position
function generateFood() {
    do {
        food = {
            x: Math.floor(Math.random() * WIDTH),
            y: Math.floor(Math.random() * HEIGHT)
        };
    } while (snake.some(seg => seg.x === food.x && seg.y === food.y));
}

// Move snake
function moveSnake() {
    if (gamePaused) return;
    
    const head = {
        x: snake[0].x + direction.x,
        y: snake[0].y + direction.y
    };
    
    // Check wall collision
    if (head.x < 0 || head.x >= WIDTH || head.y < 0 || head.y >= HEIGHT) {
        gameOver();
        return;
    }
    
    // Check self collision
    if (snake.some(seg => seg.x === head.x && seg.y === head.y)) {
        gameOver();
        return;
    }
    
    snake.unshift(head);
    
    // Check food collision
    if (head.x === food.x && head.y === food.y) {
        score += 10;
        gameSpeed = Math.max(50, INITIAL_SPEED - (score / 10) * 5); // Speed up as score increases
        generateFood();
        
        // Restart game loop with new speed
        clearInterval(gameInterval);
        gameInterval = setInterval(gameLoop, gameSpeed);
    } else {
        snake.pop();
    }
}

// Game loop
function gameLoop() {
    moveSnake();
    drawBoard();
}

// Game over
function gameOver() {
    gameRunning = false;
    clearInterval(gameInterval);
    clearScreen();
    console.log('╔' + '═'.repeat(WIDTH) + '╗');
    console.log('║' + 'GAME OVER'.padStart((WIDTH + 9) / 2).padEnd(WIDTH) + '║');
    console.log('║' + `Final Score: ${score}`.padStart((WIDTH + 13) / 2).padEnd(WIDTH) + '║');
    console.log('╚' + '═'.repeat(WIDTH) + '╝');
    console.log('\nPress R to restart or Q to quit');
}

// Restart game
function restartGame() {
    snake = [{ x: 15, y: 10 }];
    direction = { x: 1, y: 0 };
    score = 0;
    gameSpeed = INITIAL_SPEED;
    gameRunning = true;
    gamePaused = false;
    generateFood();
    clearInterval(gameInterval);
    gameInterval = setInterval(gameLoop, gameSpeed);
    drawBoard();
}

// Toggle pause
function togglePause() {
    if (!gameRunning) return;
    gamePaused = !gamePaused;
    drawBoard();
}

// Handle keyboard input
process.stdin.on('data', (key) => {
    // Handle Ctrl+C
    if (key === '\u0003') {
        process.stdout.write('\x1b[?25h'); // Show cursor
        process.exit();
    }
    
    if (!gameRunning && key.toLowerCase() !== 'r' && key.toLowerCase() !== 'q') {
        return;
    }
    
    switch(key.toLowerCase()) {
        case 'w':
        case '\u001b[A': // Up arrow
            if (direction.y !== 1) {
                direction = { x: 0, y: -1 };
            }
            break;
        case 's':
        case '\u001b[B': // Down arrow
            if (direction.y !== -1) {
                direction = { x: 0, y: 1 };
            }
            break;
        case 'a':
        case '\u001b[D': // Left arrow
            if (direction.x !== 1) {
                direction = { x: -1, y: 0 };
            }
            break;
        case 'd':
        case '\u001b[C': // Right arrow
            if (direction.x !== -1) {
                direction = { x: 1, y: 0 };
            }
            break;
        case 'p':
            togglePause();
            break;
        case 'r':
            restartGame();
            break;
        case 'q':
            process.stdout.write('\x1b[?25h'); // Show cursor
            process.exit();
            break;
    }
});

// Initialize game
generateFood();
drawBoard();
gameInterval = setInterval(gameLoop, gameSpeed);

// Cleanup on exit
process.on('SIGINT', () => {
    process.stdout.write('\x1b[?25h'); // Show cursor
    process.exit();
});

process.on('exit', () => {
    process.stdout.write('\x1b[?25h'); // Show cursor
});

