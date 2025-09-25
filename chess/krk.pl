% Hypothesis for depth 0 win in KRK end game

%:- [bk].

won_for_white((WKF, WKR, WRF, WRR, BKF, BKR)) :-
    depth_of_win(0, WKF, WKR, WRF, WRR, BKF, BKR).

depth_of_win(0, c, 2, a, A, a, 2) :-
    \+ ab3(0, c, 2, a, A, a, 2).

depth_of_win(0, c, A, a, B, a, 1) :-
    \+ ab2(0, c, A, a, B, a, 1).

depth_of_win(0, A, 3, B, 1, A, 1) :-
    \+ ab1(0, A, 3, B, 1, A, 1).

ab1(0, A, 3, B, 1, A, 1) :-
    diff(A, B, d1).

ab2(0, c, A, a, 2, a, 1).

ab3(0, c, 2, a, A, a, 2) :-
    diff(2, A, d1).


% Map chess files to numbers
file_num(a, 1).
file_num(b, 2).
file_num(c, 3).
file_num(d, 4).
file_num(e, 5).
file_num(f, 6).
file_num(g, 7).
file_num(h, 8).

% General difference of 1 predicate
diff(X, Y, d1) :-
    integer(X), integer(Y), % ranks are 1-8
    D is abs(X-Y),
    D =:= 1.

diff(X, Y, d1) :-
    atom(X), atom(Y),       % files are: a-h
    file_num(X, Xn),
    file_num(Y, Yn),
    D is abs(Xn-Yn),
    D =:= 1.

