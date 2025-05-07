#[derive(Debug)]
enum Shape {
    Rectangle { width: f64, height: f64 },
    Triangle { x: f64, y: f64, z: f64 },
}

impl Shape {
    fn square(size: f64) -> Shape {
        Shape::Rectangle {
            width: size,
            height: size,
        }
    }

    fn area(&self) -> f64 {
        match self {
            Self::Rectangle { width, height } => width * height,
            Self::Triangle { x, y, z } => {
                0.25 * ((x + y + z) * (-x + y + z) * (x - y + z) * (x + y - z)).sqrt()
            }
        }
    }

    fn can_hold(&self, other: &Shape) -> bool {
        match (self, other) {
            (
                Self::Rectangle {
                    width: w1,
                    height: h1,
                },
                Self::Rectangle {
                    width: w2,
                    height: h2,
                },
            ) => w1 >= w2 && h1 >= h2,
            (
                Self::Triangle {
                    x: x1,
                    y: y1,
                    z: z1,
                },
                Self::Triangle {
                    x: x2,
                    y: y2,
                    z: z2,
                },
            ) => x1 >= x2 && y1 >= y2 && z1 >= z2,
            _ => false,
        }
    }
}

fn main() {
    let rect1 = Shape::Rectangle {
        width: 30.0,
        height: 50.0,
    };
    let rect2 = Shape::Rectangle {
        width: 10.0,
        height: 40.0,
    };
    let rect3 = Shape::Rectangle {
        width: 60.0,
        height: 45.0,
    };
    let square1 = Shape::square(3.0);

    println!(
        "The area of the rectangle is {} square pixels.",
        rect1.area()
    );
    println!(
        "The area of the square is {} square pixels.",
        square1.area()
    );
    println!("Can rect1 hold rect2? {}", rect1.can_hold(&rect2));
    println!("Can rect1 hold rect3? {}", rect1.can_hold(&rect3));

    let traingle1 = Shape::Triangle {
        x: 2.0,
        y: 2.0,
        z: 2.0,
    };
    let traingle2 = Shape::Triangle {
        x: 1.0,
        y: 1.0,
        z: 1.0,
    };
    let traingle3 = Shape::Triangle {
        x: 1.2,
        y: 3.0,
        z: 1.0,
    };

    println!(
        "The area of the triangle is {} square pixels.",
        traingle1.area()
    );
    println!(
        "The area of the triangle is {} square pixels.",
        traingle2.area()
    );
    println!(
        "Can traingle1 hold traingle2? {}",
        traingle1.can_hold(&traingle2)
    );
    println!(
        "Can traingle1 hold traingle3? {}",
        traingle1.can_hold(&traingle3)
    );
}
