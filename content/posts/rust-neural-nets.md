+++
title = "neural nets from scratch in rust"
date = "2026-01-09T20:38:58Z"
author = "josh"
tags = ["ml"]
description = "cracked devs on twitter are writing ML libraries from scratch in C"
showFullContent = false
readingTime = true
+++

its easy to make pytorch go brr and overfit MNIST with two hands tied behind your back, so i thought i would make it harder and overfit MNIST in my own library in rust, using nothing but the `std` library.

**warning: i am not a rust developer, or even a low level developer at all, and have very little idea what i am doing.**

# the matrix

![neo](/images/neo.png)

karpathy's already done the [micrograd thing](https://github.com/karpathy/micrograd), where gradients and differentiation are done on the scalar value level, and i dont want to just carbon copy that, but i also dont want to do the full shebang with tensor ops, so i thought id meet in the middle with simple 2d matrices.

a matrix is just a collection of values (lets assume f32 globally), stored in rows and columns.

coming from python/numpy, matrix rows are just lists of values, and a matrix is just a list of rows, so it would be easy to assume that a matrix should hold this structure in memory.

```python
my_nice_matrix = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
])
```
```rust
pub struct Matrix {
    nrows: usize,
    ncols: usize,
    data: Vec<Vec<f32>>
}
impl Matrix {
    pub fn get(&self, row: usize, col: usize) -> f32 {
        assert!(row < self.nrows, "row out of bounds");
        assert!(col < self.ncols, "col out of bounds");
        self.data[row][col]
    }
}
```

although this library isn't going to be topping any benchmarks any time soon, i already know this is setting us up for bad performance. a rust `Vec` is a heap allocated, growable array, and the program sees it as 3 components on the stack:

- a pointer to the data on the heap
- how long the data currently is
- what capacity the data is allowed to have

so if we declare a `Vec<Vec<f32>>`, we're going to have pointers pointing to pointers pointing to data, which even to my small python brain seems inefficient.

a better representation is keeping one `Vec` that has all the data nicely arranged, and we just handle the indexing algebra ourselves.

```rust
pub struct Matrix {
    nrows: usize,
    ncols: usize,
    pub data: Vec<f32>,
}
impl Matrix {
    pub fn get(&self, row: usize, col: usize) -> f32 {
        assert!(row < self.nrows, "row out of bounds");
        assert!(col < self.ncols, "col out of bounds");
        self.data[row * self.ncols + col]
    }
}
```

next we need those nice matrix operations that let us do things we're likely going to want to do, primarily matrix multiplication, transposition, and simple elementwise operations.

```rust
impl Matrix {

    pub fn add(&self, rhs: &Matrix) -> Matrix {
        assert!(
            self.shape() == rhs.shape(),
            "shape mismatch in add"
        );

        let mut data = Vec::with_capacity(self.data.len());
        for i in 0..self.data.len() {
            data.push(self.data[i] + rhs.data[i]);
        }

        Matrix {
            nrows: self.nrows,
            ncols: self.ncols,
            data,
        }
    }

    pub fn transpose(&self) -> Matrix {
        let mut data = vec![0.0; self.nrows * self.ncols];

        for r in 0..self.nrows {
            for c in 0..self.ncols {
                data[c * self.nrows + r] =
                    self.data[r * self.ncols + c];
            }
        }

        Matrix {
            nrows: self.ncols,
            ncols: self.nrows,
            data,
        }
    }
}
```

allocating new memory for outputs is not the most efficient way of doing this i'm sure, but we're just going to `--release` and cross our fingers and pray at the end.

overall, elementwise operations are easy and transposition is easy with a bit of thought, but lets flesh out the matrix multiplication for the sake of trying to remember my uni linear algebra.

the elements of a product of two matrices $X \in \mathbb{R}^{n \times m}$ and $Y \in \mathbb{R}^{m \times p}$ are given by:
$(XY)_{ij} = \sum_{k=1}^{m}X_{ik}Y_{kj}$

i.e. for fixed $(i,j)$, we introduce a summation variable $k$ to dot product the row vectors of $X$ and the column vectors of $Y$.

this summation variable means matrix multiplication is an $O(nmp)$ operation (or $O(n^3)$ for square matrices), making it expensive unless you are Jensen Huang.

```rust
    pub fn matmul(&self, rhs: &Matrix) -> Matrix {
        assert!(
            self.ncols == rhs.nrows,
            "shape mismatch in matmul"
        );

        let mut out = vec![0.0; self.nrows * rhs.ncols];

        for i in 0..self.nrows {
            for k in 0..self.ncols {
                let a = self.data[i * self.ncols + k];
                let rhs_row = k * rhs.ncols;
                let out_row = i * rhs.ncols;

                for j in 0..rhs.ncols {
                    out[out_row + j] += a * rhs.data[rhs_row + j];
                }
            }
        }

        Matrix {
            nrows: self.nrows,
            ncols: rhs.ncols,
            data: out,
        }
    }
```

the order of the loops in the summation is also somewhat important, due to how we decide to index our underlying data vector. because we are indexing rows before columns, we are assuming our data is in a *row-major* format, so looping over the k before the j allows the cpu to have a more contiguous view of the data in memory (something something cache locality).

now we've got the simple ops over, we can start with the AI.

# the neural bit

if you're a nerd then you can get some botched maths in a separate post, [backpropagation for dorks](/posts/backpropagation-math/).

tldr: neural networks are a linear transformation (matrix multiplication) plus a constant bias, all pumped through a differentiable non-linear function, then we gradually push the values we're transforming by in the direction that will improve our outputs.

if we have lots of linear transformations and non-linear functions (activation functions) in a row, we can differentiate it all at once via backpropagation using the chain rule

$$\frac{\partial L}{\partial X} = \frac{\partial L}{\partial W_1} \frac{\partial W_1}{\partial W_2} ... \frac{\partial W_k}{\partial X}$$

so what does this look like in code?


