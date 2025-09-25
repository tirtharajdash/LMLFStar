% krk_legal_checkmate.pl
% Position encoding: (WKF, WKR, WRF, WRR, BKF, BKR)
% Files are atoms a..h, ranks are integers 1..8.

%%%%%%% Board primitives

file(a). file(b). file(c). file(d). file(e). file(f). file(g). file(h).
rank(1). rank(2). rank(3). rank(4). rank(5). rank(6). rank(7). rank(8).

square(F,R) :- file(F), rank(R).

file_idx(a,1). file_idx(b,2). file_idx(c,3). file_idx(d,4).
file_idx(e,5). file_idx(f,6). file_idx(g,7). file_idx(h,8).

abs_diff(X,Y,D) :- (X>=Y -> D is X-Y ; D is Y-X).

same_square(F1,R1,F2,R2) :- F1==F2, R1==R2.

king_adjacent(F1,R1,F2,R2) :-
    file_idx(F1,I1), file_idx(F2,I2),
    abs_diff(I1,I2,DX), abs_diff(R1,R2,DY),
    DX =< 1, DY =< 1, \+ same_square(F1,R1,F2,R2).

%%%%%%% Occupancy helpers

empty_between_file(F, R1, R2, (WKF,WKR,WRF,WRR,BKF,BKR)) :-
    (R1<R2 -> Lo is R1+1, Hi is R2-1 ; Lo is R2+1, Hi is R1-1),
    forall(between(Lo,Hi,R),
           (\+ same_square(WKF,WKR,F,R),
            \+ same_square(WRF,WRR,F,R),
            \+ same_square(BKF,BKR,F,R))).

empty_between_rank(R, F1, F2, (WKF,WKR,WRF,WRR,BKF,BKR)) :-
    file_idx(F1,I1), file_idx(F2,I2),
    (I1<I2 -> Lo is I1+1, Hi is I2-1 ; Lo is I2+1, Hi is I1-1),
    forall(between(Lo,Hi,I),
           ( file_idx(F,I),
             \+ same_square(WKF,WKR,F,R),
             \+ same_square(WRF,WRR,F,R),
             \+ same_square(BKF,BKR,F,R))).

%%%%%%% Attack relations

king_attacks(F1,R1,F2,R2) :- king_adjacent(F1,R1,F2,R2).

rook_attacks(WRF,WRR,TF,TR, Pos) :-
    ( WRF == TF,
      empty_between_file(WRF, WRR, TR, Pos)
    ; WRR == TR,
      empty_between_rank(WRR, WRF, TF, Pos)
    ).

white_attacks_square((WKF,WKR,WRF,WRR,_BKF,_BKR), (TF,TR)) :-
    king_attacks(WKF,WKR,TF,TR)
 ;  rook_attacks(WRF,WRR,TF,TR, (WKF,WKR,WRF,WRR,_BKF,_BKR)).

% After BK captures the rook, only the white king’s attacks remain.
white_attacks_square_after_capture((WKF,WKR,_,_,_,_), (TF,TR)) :-
    king_attacks(WKF,WKR,TF,TR).

%%%%%%% Position legality (KRK constraints)

legal_position_krk((WKF,WKR,WRF,WRR,BKF,BKR)) :-
    square(WKF,WKR), square(WRF,WRR), square(BKF,BKR),
    \+ same_square(WKF,WKR,WRF,WRR),
    \+ same_square(WKF,WKR,BKF,BKR),
    \+ same_square(WRF,WRR,BKF,BKR),
    \+ king_adjacent(WKF,WKR,BKF,BKR).

%%%%%%% Check / Checkmate (Black to move)

black_in_check((WKF,WKR,WRF,WRR,BKF,BKR)) :-
    white_attacks_square((WKF,WKR,WRF,WRR,BKF,BKR), (BKF,BKR)).

black_safe_square((WKF,WKR,WRF,WRR,_BKF,_BKR), (NF,NR)) :-
    square(NF,NR),
    \+ same_square(WKF,WKR,NF,NR),
    \+ king_adjacent(WKF,WKR,NF,NR),
    ( same_square(WRF,WRR,NF,NR) ->
        \+ white_attacks_square_after_capture((WKF,WKR,WRF,WRR,_BKF,_BKR), (NF,NR))
    ;
        \+ white_attacks_square((WKF,WKR,WRF,WRR,_BKF,_BKR), (NF,NR))
    ).

black_legal_king_move((WKF,WKR,WRF,WRR,BKF,BKR), (NF,NR)) :-
    file_idx(BKF,I1), file_idx(NF,I2),
    abs_diff(I1,I2,DX), abs_diff(BKR,NR,DY),
    DX =< 1, DY =< 1,
    \+ same_square(BKF,BKR,NF,NR),
    \+ same_square(WKF,WKR,NF,NR),
    black_safe_square((WKF,WKR,WRF,WRR,BKF,BKR), (NF,NR)).

black_has_legal_move((WKF,WKR,WRF,WRR,BKF,BKR)) :-
    member(Df, [-1,0,1]),
    member(Dr, [-1,0,1]),
    (Df \= 0 ; Dr \= 0),
    file_idx(BKF,If), I2 is If + Df,
    between(1,8,I2), file_idx(NF,I2),
    NR is BKR + Dr, rank(NR),
    black_legal_king_move((WKF,WKR,WRF,WRR,BKF,BKR), (NF,NR)), !.

checkmate_krk((WKF,WKR,WRF,WRR,BKF,BKR)) :-
    legal_position_krk((WKF,WKR,WRF,WRR,BKF,BKR)),
    black_in_check((WKF,WKR,WRF,WRR,BKF,BKR)),
    \+ black_has_legal_move((WKF,WKR,WRF,WRR,BKF,BKR)).

