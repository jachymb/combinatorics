node(1..n).
0 { edge(X, Y)  } 1 :- node(X), node(Y), X < Y.
1 { color(X, 1..coloredness) } 1 :- node(X).
edge(X, Y) :- edge(Y, X).
:- color(X, C), color(Y, C), edge(X, Y).
:- #count{ Y : edge(X, Y) } = N, node(X), N != regularity.
