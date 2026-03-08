use std::fmt;

const LIMIT: i32 = 10;

struct Counter {
    value: i32,
}

impl Counter {
    fn current(&self) -> i32 {
        self.value
    }

    fn inc(&mut self, delta: i32) {
        self.value += delta;
    }
}

fn build(limit: i32, offset: i32) -> i32 {
    limit + offset
}
