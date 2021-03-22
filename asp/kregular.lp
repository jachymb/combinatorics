node(1..n).
0 { edge(X, Y)  } 1 :- node(X), node(Y), X < Y.
edge(X, Y) :- edge(Y, X).
:- #count{ Y : edge(X, Y) } = N, node(X), N != k.
