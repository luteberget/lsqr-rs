# Sparse least squares

This is a port from C code from [web.stanford.edu/group/SOL/software/lsqr](https://web.stanford.edu/group/SOL/software/lsqr/). It is a conjugate-gradient method for solving sparse linear equations and sparse least-squares problems. It solves `Ax = b`, or minimizes `|Ax-b|^2`, or minimizes the damped form `|Ax-b|^2 + l^2*|x|^2`. See the original web page for more details.

## TODO

 * [ ] Take RHS array by move, as it is has unspecified contents afterwards.
 * [ ] Take an initial value as parameter.

## Usage

The Rust API is single function `lsqr` which takes the size of the matrix, 
an initial suggestion for the solution vector, and a function that
the solver can call to update `y = y + A * x` or `x = x + A^T * y`, 
leaving the representation of `A` up to the caller.

Here is an example demonstrating the principle of calculating the required expressions,
but note that this is not actually a sparse representation, and in this case you might
be better off with a dense solver such as LAPACK (see [netlib LLS](https://www.netlib.org/lapack/lug/node27.html)).

```rust
let params = Params {
    damp :0.0,         // Damping factor -- for miniminizing |Ax-b|^2 + damp^2 * x^2.
    rel_mat_err :1e-6, // Estimated relative error in the data defining the matrix A.
    rel_rhs_err :1e-6, // Estimated relative error in the right-hand side vector b.
    condlim :0.0,      // Upper limit on the condition number of A_bar (see original source code).
    iterlim :100,      // Limit on number of iterations
};

let mut rhs = vec![-1.,7.,2.];
let matrix = vec![3.,4.,0.,-6.,-8.,1.];
let n_rows = 3; let n_cols = 2;

let aprod = |mode :Product| {
    match mode {
	Product::YAddAx { x, y } =>  {
	    // y += A*x   [m*1] = [m*n]*[n*1]
	    for i in 0..n_rows {
		for j in 0..n_cols {
		    y[i] += matrix[n_rows*j + i] * x[j];
		}
	    }
	},
	Product::XAddATy { x, y } => {
	    // x += A^T*y  [n*1] = [n*m][m*1]
	    for i in 0..n_cols {
		for j in 0..n_rows {
		    x[i] += matrix[n_rows*i + j] * y[j];
		}
	    }
	},
    };
};

let (sol,statistics) = lsqr(|msg| print!("{}", msg), 
                            n_rows, n_cols, params, aprod, &mut rhs);
```

