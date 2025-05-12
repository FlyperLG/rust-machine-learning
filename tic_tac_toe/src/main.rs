use rand::Rng;
use std::io;

struct Grid {
    grid: [[char; 3]; 3],
}

impl Grid {
    fn print(&self) {
        println!("\n  A   B   C");
        for (index, row) in self.grid.iter().enumerate() {
            println!("{} {} | {} | {}", index + 1, row[0], row[1], row[2]);
            if index < row.len() - 1 {
                println!(" -----------");
            };
        }
    }

    fn set_value(&mut self, row: usize, col: usize, char: char) -> bool {
        if row > 2 || col > 2 || self.grid[row][col] != '-' {
            false
        } else {
            self.grid[row][col] = char;
            true
        }
    }

    fn check_game_over(&self) -> bool {
        for row in self.grid {
            let mut has_row: bool = true;
            let mut row_value: char = '-';
            for value in row {
                if value == '-' {
                    has_row = false;
                    break;
                }

                if value != row_value && row_value != '-' {
                    has_row = false;
                    break;
                }

                row_value = value;
            }

            if has_row {
                return true;
            }
        }

        for col_index in 0..3 {
            let mut has_col: bool = true;
            let mut col_value: char = '-';
            for row_index in 0..3 {
                let value = self.grid[row_index][col_index];
                if value == '-' {
                    has_col = false;
                    break;
                }

                if value != col_value && col_value != '-' {
                    has_col = false;
                    break;
                }

                col_value = value;
            }
            if has_col {
                return true;
            }
        }

        let mut has_diagonal: bool = true;
        let mut diagonal_value: char = '-';
        for diagonal_index in 0..3 {
            let value = self.grid[diagonal_index][diagonal_index];
            if value == '-' {
                has_diagonal = false;
                break;
            }

            if value != diagonal_value && diagonal_value != '-' {
                has_diagonal = false;
                break;
            }

            diagonal_value = value;
        }
        if has_diagonal {
            return true;
        }

        let mut has_diagonal: bool = true;
        let mut diagonal_value: char = '-';
        for diagonal_index in 0..3 {
            let value = self.grid[0 + diagonal_index][2 - diagonal_index];
            if value == '-' {
                has_diagonal = false;
                break;
            }

            if value != diagonal_value && diagonal_value != '-' {
                has_diagonal = false;
                break;
            }

            diagonal_value = value;
        }
        if has_diagonal {
            return true;
        }
        return false;
    }
}

enum Player {
    ConsolePlayer { name: String, display_char: char },
    RandomPlayer { name: String, display_char: char },
}

impl Player {
    fn get_position(&self) -> Option<(usize, usize)> {
        match self {
            Self::ConsolePlayer { name, .. } => {
                println!("{} please select a row by typing 1/2/3.", name);
                let mut row: String = String::new();
                io::stdin()
                    .read_line(&mut row)
                    .expect("Failed to read line");
                let row: usize = match row.trim().parse::<usize>() {
                    Ok(num) => num - 1,
                    Err(_) => return None,
                };

                println!("{} please select a col by typing A/B/C.", name);
                let mut col: String = String::new();
                io::stdin()
                    .read_line(&mut col)
                    .expect("Failed to read line");
                let col: usize = match col.trim().to_uppercase().as_str() {
                    "A" => 0,
                    "B" => 1,
                    "C" => 2,
                    _ => return None,
                };
                return Some((row, col));
            }
            Self::RandomPlayer {
                name: _,
                display_char: _,
            } => {
                let row: usize = rand::rng().random_range(0..=2);
                let col: usize = rand::rng().random_range(0..=2);
                return Some((row, col));
            }
        };
    }

    fn display_char(&self) -> char {
        match self {
            Self::ConsolePlayer { display_char, .. } => *display_char,
            Self::RandomPlayer {
                name: _,
                display_char,
            } => *display_char,
        }
    }

    fn name(&self) -> &str {
        match self {
            Self::ConsolePlayer { name, .. } => name,
            Self::RandomPlayer {
                name,
                display_char: _,
            } => name,
        }
    }
}

fn main() {
    let mut grid = Grid {
        grid: [['-'; 3]; 3],
    };

    let players = [
        Player::ConsolePlayer {
            name: String::from("Player1"),
            display_char: 'X',
        },
        Player::RandomPlayer {
            name: String::from("Player2"),
            display_char: 'O',
        },
    ];

    let mut player_index = 0;
    loop {
        grid.print();
        let player = &players[player_index];

        loop {
            if let Some((row, col)) = player.get_position() {
                if grid.set_value(row, col, player.display_char()) {
                    break;
                }
            }

            if let Player::ConsolePlayer {
                name,
                display_char: _,
            } = player
            {
                println!("{} please select a valid field.", name);
            }
        }

        let is_game_over = grid.check_game_over();
        if is_game_over {
            grid.print();
            println!("{} has won the game.", player.name());
            break;
        }
        player_index += 1;
        player_index = player_index % 2;
    }
}
