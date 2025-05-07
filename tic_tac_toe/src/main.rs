use std::io;

struct Grid {
    grid: [[char; 3]; 3],
}

impl Grid {
    fn print(&self) {
        println!("  A   B   C");
        for (index, row) in self.grid.iter().enumerate() {
            println!("{} {} | {} | {}", index + 1, row[0], row[1], row[2]);
            if index < row.len() - 1 {
                println!(" -----------");
            };
        }
    }

    fn set_value(&mut self, row: usize, col: usize, char: char) {
        self.grid[row][col] = char;
    }
}

struct Player {
    name: String,
    display_char: char,
}

struct InputManager;
impl InputManager {
    fn get_position(player: &Player) -> Option<(usize, usize)> {
        println!("{} please select a row by typing 1/2/3.", player.name);
        let mut row: String = String::new();
        io::stdin()
            .read_line(&mut row)
            .expect("Failed to read line");
        let row: usize = match row.trim().parse::<usize>() {
            Ok(num) => num - 1,
            Err(_) => return None,
        };

        println!("{} please select a col by typing 1/2/3.", player.name);
        let mut col: String = String::new();
        io::stdin()
            .read_line(&mut col)
            .expect("Failed to read line");
        let col: usize = match col.trim().parse::<usize>() {
            Ok(num) => num - 1,
            Err(_) => return None,
        };
        return Some((row, col));
    }
}

fn main() {
    let mut grid = Grid {
        grid: [['-'; 3]; 3],
    };
    let player = Player {
        name: String::from("Player1"),
        display_char: 'X',
    };

    loop {
        grid.print();
        match InputManager::get_position(&player) {
            Some((row, col)) => grid.set_value(row, col, player.display_char),
            None => continue,
        }
    }
}
