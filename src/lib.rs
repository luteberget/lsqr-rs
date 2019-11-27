//! # Sparse least squares
//! 
//! This is a port from C code from [web.stanford.edu/group/SOL/software/lsqr](https://web.stanford.edu/group/SOL/software/lsqr/). It is a conjugate-gradient method for solving sparse linear equations and sparse least-squares problems. It solves `Ax = b`, or minimizes `|Ax-b|^2`, or minimizes the damped form `|Ax-b|^2 + l^2*|x|^2`. See the original web page for more details.
//! 
//! The Rust API is single function `lsqr` which takes the size of the matrix, 
//! an initial suggestion for the solution vector, and a function that
//! the solver can call to update `y = y + A * x` or `x = x + A^T * y`, 
//! leaving the representation of `A` up to the caller.
//! 
//! Here is an example demonstrating the principle of calculating the required expressions,
//! but note that this is not actually a sparse representation, and in this case you might
//! be better off with a dense solver such as LAPACK (see [netlib LLS](https://www.netlib.org/lapack/lug/node27.html)).
//! 
//! ```rust
//! let params = lsqr::Params {
//!     damp :0.0,         // Damping factor -- for miniminizing |Ax-b|^2 + damp^2 * x^2.
//!     rel_mat_err :1e-6, // Estimated relative error in the data defining the matrix A.
//!     rel_rhs_err :1e-6, // Estimated relative error in the right-hand side vector b.
//!     condlim :0.0,      // Upper limit on the condition number of A_bar (see original source code).
//!     iterlim :100,      // Limit on number of iterations
//! };
//! 
//! let mut rhs = vec![-1.,7.,2.];
//! let matrix = vec![3.,4.,0.,-6.,-8.,1.];
//! let n_rows = 3; let n_cols = 2;
//! 
//! let aprod = |mode :lsqr::Product| {
//!     match mode {
//! 	lsqr::Product::YAddAx { x, y } =>  {
//! 	    // y += A*x   [m*1] = [m*n]*[n*1]
//! 	    for i in 0..n_rows {
//! 		for j in 0..n_cols {
//! 		    y[i] += matrix[n_rows*j + i] * x[j];
//! 		}
//! 	    }
//! 	},
//! 	lsqr::Product::XAddATy { x, y } => {
//! 	    // x += A^T*y  [n*1] = [n*m][m*1]
//! 	    for i in 0..n_cols {
//! 		for j in 0..n_rows {
//! 		    x[i] += matrix[n_rows*i + j] * y[j];
//! 		}
//! 	    }
//! 	},
//!     };
//! };
//! 
//! let (sol,statistics) = lsqr::lsqr(|msg| print!("{}", msg), 
//!                                   n_rows, n_cols, params, aprod, &mut rhs);
//! ```

/// Resulting state after executing `lsqr`.
#[derive(Debug,Copy,Clone)]
pub enum ResultMsg {
    /// The initial guess was exactly correct.
    OkInputIsExact,
    /// `Ax - b` is within given tolerances.
    OkResidualAtol,
    /// The least-squares solution is within given tolerances.
    OkErrorAtol,
    /// The estimate of the condition number of A_bar was exceeded.
    OkCondnumLimit,
    /// `Ax - b` is within machine precision.
    OkResidualEps,
    /// The estimate of the condition number of A_bar is too large.
    OkErrorEps,
    OkCondnumEps,
    ErrIterLim,
}

/// Numerical parameters for `lsqr`.
#[derive(Debug,Copy,Clone)]
pub struct Params {
    /// Damping factor -- for miniminizing |Ax-b|^2 + damp^2 * x^2.
    pub damp :f64,
    /// Estimated relative error in the data defining the matrix A.
    pub rel_mat_err :f64,
    /// Estimated relative error in the right-hand side vector b.
    pub rel_rhs_err :f64,
    /// Upper limit on the condition number of A_bar (see original source code).
    pub condlim :f64,
    /// Limit on number of iterations
    pub iterlim :usize,
}

impl Default for Params {
    fn default() -> Self {
        Params {
            damp: 0.0,
            rel_mat_err: 0.0,
            rel_rhs_err: 0.0,
            condlim: 0.0,
            iterlim: 300000,
        }
    }
}

/// Numerical statistics from the execution of `lsqr`.
#[derive(Debug,Copy,Clone)]
pub struct Statistics {
    /// Termination message -- see `ResultMsg`.
    pub term_flag :ResultMsg,
    /// Total number of iterations performed.
    pub num_iters: usize,
    /// An estimate of the Froebnius norm of A_bar. (See the original source)
    pub frob_mat_norm :f64,
    /// An estimate of the conditioning number of A_bar. (See the original source)
    pub mat_cond_num :f64,
    /// An estimate of the final RHS residual. (See the original source)
    pub resid_norm :f64,
    /// An estimate of the final matrix residual. (See the original source)
    pub mat_resid_norm :f64,
    /// An estiamte of the norm of the final solution vector. (See the original source)
    pub sol_norm :f64,
}

/// Parameters passed to the user-supplied function parameter to `lsqr`.
pub enum Product<'a> {
    /// Compute y = y + A * x
    YAddAx {
        y :&'a mut [f64],
        x :&'a [f64],
    },

    /// Compute x = x + A.transposed * y
    XAddATy {
        x :&'a mut [f64],
        y :&'a [f64],
    }
}

/// A conjugate-gradient method for solving sparse linear equations and sparse least-squares problems. 
/// It solves `Ax = b`, or minimizes `|Ax-b|^2`, or minimizes the damped form `|Ax-b|^2 + l^2*|x|^2`. 
/// See the [original web page](https://web.stanford.edu/group/SOL/software/lsqr/) for more details.
pub fn lsqr(mut log :impl FnMut(&str),
            rows :usize, 
            cols :usize, 
            params :Params,
            mut aprod :impl FnMut(Product),
            rhs :&mut [f64]) -> (Vec<f64>, Statistics) {

    log(         "  Least Squares Solution of A*x = B\n");
    log(&format!("    The matrix A has {} rows and {} columns\n", rows, cols));
    log(&format!("    The damping parameter is DAMP = {}\n", params.damp));
    log(&format!("    ATOL = {}\t\tCONDLIM = {}\n", params.rel_mat_err, params.condlim));
    log(&format!("    BTOL = {}\t\tITERLIM = {}\n", params.rel_mat_err, params.iterlim));

    fn norm2(x :&[f64]) -> f64 { x.iter().map(|e| e*e).sum::<f64>().sqrt() }
    fn scale(s :f64, x :&mut [f64]) { for e in x.iter_mut() { *e *= s; } }

    let rel_mat_err = params.rel_mat_err;
    let rel_rhs_err = params.rel_rhs_err;

    let mut num_iters = 0;

    let mut frob_mat_norm = 0.0;
    let mut mat_cond_num = 0.0;
    let mut sol_norm = 0.0;

    let mut bbnorm = 0.0;
    let mut ddnorm = 0.0;
    let mut xxnorm = 0.0;
    let mut cs2 = -1.0;
    let mut sn2 = 0.0;
    let mut zeta = 0.0;
    let mut res = 0.0;

    let mut psi;

    let cond_tol = if params.condlim > 0.0 { 1.0 / params.condlim } else { std::f64::EPSILON };
    
    let mut alpha = 0.0;
    let mut beta;


    // Set up the initial vectors u and v for bidiagonalization. These satify the relations
    //   BETA * u = b - A * x0
    //   ALPHA * v = A^T * u
    //

    let mut sol_vec = vec![0.0; cols];
    let mut bidiag_work = vec![0.0; cols];

    // Compute b - A * x0 and store in vector u which initially held vector b.
    scale(-1.0, rhs);
    aprod(Product::YAddAx { y: rhs, x: &sol_vec });
    scale(-1.0, rhs);

    // Compute Euclidean length of u and store as beta.
    beta = norm2(rhs);

    if beta > 0.0 {
        // scale u by inverse of beta
        scale(1.0 / beta, rhs);
        // copmute matrix-vector product A^T * u and store it in vector v.
        //aprod(Mode::Two, &mut bidiag_work, rhs);
        aprod(Product::XAddATy { x: &mut bidiag_work, y: rhs });
        // Compute Euclidean length of v and store as alpha
        alpha = norm2(&mut bidiag_work);
    }

    if alpha > 0.0 {
        // Scale vector v by the inverse of alpha
        scale(1.0 / alpha, &mut bidiag_work);
    }

    let mut srch_dir_work = bidiag_work.clone();

    let mut mat_resid_norm = alpha * beta;
    let mut resid_norm = beta;
    let bnorm = beta;


    // If the norm || A^T r || is zero, then the inital guess is the exact
    // solution. Exit and report.
    if mat_resid_norm == 0.0 {
        log("early exit resid norm");
        return (sol_vec, Statistics {
            term_flag: ResultMsg::OkInputIsExact,
            num_iters: num_iters,
            frob_mat_norm: frob_mat_norm,
            mat_cond_num: mat_cond_num,
            resid_norm :resid_norm,
            mat_resid_norm: mat_resid_norm,
            sol_norm: sol_norm,
        });
    }

    let mut rhobar = alpha;
    let mut phibar = beta;


    // Print iteration header.
    log("first iteration");


    // The main iteration loop is continued as long as no stopping criteria are
    // satisfied and the number of total iterations is less than some upper bound.
    let term_flag = loop {
        num_iters += 1;

        // Perform the next step of the bidiagonalization to obtain the next
        // vectors u and v, and the scalars alpha and beta.
        // These satisfy the relations
        //    BETA * u = A * v - ALPHA * u
        //    ALPHA * v = A^T * u - BETA * v

        // Scale vector u by the negative of alpha
        scale(-alpha, rhs);
        // Compute A*v - ALPÅHA*u and store in vector u
        //aprod(Mode::One, &mut bidiag_work, rhs);
        aprod(Product::YAddAx { y: rhs, x: &bidiag_work });
        // Compute Euclidean length of u and store as BETA
        beta = norm2(rhs);

        // Accumulate this quatity to estimate froebnius norm of matrix A
        bbnorm += alpha*alpha + beta*beta + params.damp*params.damp;

        if beta > 0.0 {
            scale(1.0 / beta, rhs);
            scale(-beta, &mut bidiag_work);
            // Compute A^T * u - BETA * v and store in vector v
            //aprod(Mode::Two, &mut bidiag_work, rhs);
            aprod(Product::XAddATy { x: &mut bidiag_work, y: rhs });
            alpha = norm2(&bidiag_work);
            if alpha > 0.0 {
                scale(1.0 / alpha, &mut bidiag_work);
            }
        }

        // Use a plane rotation to eliminate the damping parameter.
        // This alters the diagonal (RHOBAR) of the lower-bidiagonal matrix
        let cs1 = rhobar / (rhobar*rhobar + params.damp*params.damp).sqrt();
        let sn1 = params.damp / (rhobar*rhobar + params.damp*params.damp).sqrt();
        psi = sn1 * phibar;
        phibar = cs1*phibar;
        
        // Use a plane rotation to eliminate the subdiagonal element (BETA)
        // of the lower-bidiagonal matrix, giving an upper-bidiagonal matrix.

        let rho = (rhobar*rhobar + params.damp*params.damp + beta*beta).sqrt();
        let cs = (rhobar*rhobar + params.damp*params.damp).sqrt() / rho;
        let sn = beta / rho;
        let theta = sn*alpha;
        rhobar = -cs * alpha;
        let phi = cs*phibar;
        phibar = sn*phibar;
        let tau = sn*phi;


        // Update solution vector
        for i in 0..cols {
            sol_vec[i] += (phi / rho) * srch_dir_work[i];
            // std err [i] = ...
            ddnorm += ( 1.0 / rho * srch_dir_work[i] ) * 
                      ( 1.0 / rho * srch_dir_work[i] );
            srch_dir_work[i] = bidiag_work[i] - (theta/rho)*srch_dir_work[i];
        }

        // Use a plane rotation ont he right to eliminate the super-diagonal element
        // (THETA) of the upper-bidiagnoal matrix. Then use the result to estimate
        // the solution norm || x ||.
        //



        let delta = sn2*rho;
        let gammabar = -cs2*rho;
        let zetabar = (phi - delta*zeta)/gammabar;

        sol_norm = (xxnorm + zetabar*zetabar).sqrt();

        let gamma = (gammabar*gammabar + theta*theta).sqrt();
        cs2 = gammabar / gamma;
        sn2 = theta / gamma;
        zeta = (phi - delta*zeta) / gamma;
        xxnorm += zeta*zeta;

        frob_mat_norm = bbnorm.sqrt();
        mat_cond_num = frob_mat_norm * ddnorm.sqrt();
        res += psi*psi;

        resid_norm = (phibar*phibar + res).sqrt();
        mat_resid_norm = alpha * tau.abs();

        // Use these norms to estimate the values of the three stopping criteria

        let stop_crit_1 = resid_norm / bnorm;
        //println!("stop crit 1 = resid norm {}  / bnorm {}  = {} ", resid_norm, bnorm, stop_crit_1);

        let mut stop_crit_2 = 0.0;
        if resid_norm > 0.0 {
            stop_crit_2 = mat_resid_norm / (frob_mat_norm*resid_norm);
        }

        let stop_crit_3 = 1.0 / mat_cond_num;

        let resid_tol = rel_rhs_err + rel_mat_err*frob_mat_norm*sol_norm / bnorm;
        let resid_tol_mach = std::f64::EPSILON + std::f64::EPSILON*
            frob_mat_norm * sol_norm / bnorm;

        //println!("resid tol = btol={}  +  atol={} * frob_mat_norm={} * sol_norm={} / bnorm = {}    = {}",
                 //rel_rhs_err, rel_mat_err, frob_mat_norm, sol_norm, bnorm, resid_tol);

        // TERMINATE
        //

        if num_iters > params.iterlim { break ResultMsg::ErrIterLim; }
        if stop_crit_3 <= std::f64::EPSILON { break ResultMsg::OkCondnumEps; }
        if stop_crit_2 <= std::f64::EPSILON { break ResultMsg::OkErrorEps; }
        if stop_crit_1 <= resid_tol_mach { break ResultMsg::OkResidualEps; }
        if stop_crit_3 <= cond_tol { break ResultMsg::OkCondnumLimit; }
        if stop_crit_2 <= rel_mat_err { break ResultMsg::OkErrorAtol; }
        if stop_crit_1 <= resid_tol { break ResultMsg::OkResidualAtol; }

        // TODO requrire two consecutive rounds

        log("iteration data\n");
    };


    // TODO Standard error estimates output

    let statistics = Statistics {
        term_flag: term_flag,
        num_iters: num_iters,
        frob_mat_norm :frob_mat_norm,
        mat_cond_num :mat_cond_num,
        resid_norm :resid_norm,
        mat_resid_norm :mat_resid_norm,
        sol_norm :sol_norm,
    };

    (sol_vec, statistics)
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {

        let params = Params {
            damp :0.0,
            rel_mat_err :1e-6,
            condlim :0.0,
            rel_rhs_err :1e-6,
            iterlim :100,
        };

        let mut rhs = vec![-1.,7.,2.];
        let matrix = vec![3.,4.,0.,-6.,-8.,1.];
        let rows = 3; let cols = 2;

        let aprod = |mode :Product| {
            match mode {
                Product::YAddAx { x, y } =>  {
                    // y += A*x   [m*1] = [m*n]*[n*1]
                    for i in 0..rows {
                        for j in 0..cols {
                            y[i] += matrix[rows*j + i] * x[j];
                        }
                    }
                },
                Product::XAddATy { x, y } => {
                    // x += A^T*y  [n*1] = [n*m][m*1]
                    for i in 0..cols {
                        for j in 0..rows {
                            x[i] += matrix[rows*i + j] * y[j];
                        }
                    }
                },
            };
        };

        let (sol,statistics) = lsqr(|msg| print!("{}", msg), rows,cols,params,aprod,&mut rhs);


        for (i,x) in sol.iter().enumerate() {
            println!("x[{}] = {:.4}", i, x);
        }
        println!("Statistics {:#?}", statistics);
    }

    #[test]
    fn test2() {
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

let (_sol,_statistics) = lsqr(|msg| print!("{}", msg), 
                            n_rows, n_cols, params, aprod, &mut rhs);
    }
}
