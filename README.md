Matrices have been generated to empasize the problem.
Their eigen values are arithmetically distributed:

sigma_i = 1 - \frac{i-1}{n-1} \times (1-\frac{1}{cond})

The generation uses QR factorization to make the matrix orthogonal
A = UDVt
Normalizing u_k:
u_k = uk - u_i \times u_i^T \times \frac{u_k}{\norm{u_k}}
