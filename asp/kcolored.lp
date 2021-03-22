node(1..n).
1 { color(X, 1..k) } 1 :- node(X).
0 { edge(X, Y)  } 1 :- node(X), node(Y), X < Y.
edge(X, Y) :- edge(Y, X).

:- color(X, C), color(Y, C), edge(X, Y).
