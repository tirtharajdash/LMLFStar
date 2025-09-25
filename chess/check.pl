:- [krk].
%:- [iter1].
%:- [iter2].
%:- [iter3].
%:- [iter4].
%:- [iter5].
%:- [iter6].
%:- [iter7].

%:- [nosym_iter1].
%:- [nosym_iter2].
%:- [nosym_iter3].
%:- [nosym_iter4].
%:- [nosym_iter5].
%:- [nosym_iter6].

%:- [nosym_nofb_iter1].
%:- [nosym_nofb_iter2].
%:- [nosym_nofb_iter3].
%:- [nosym_nofb_iter4].
%:- [nosym_nofb_iter5].
%:- [nosym_nofb_iter6].

%:- [nosym_nofb_5c_iter1].
%:- [nosym_nofb_5c_iter2].
%:- [nosym_nofb_5c_iter3].
%:- [nosym_nofb_5c_iter4].
%:- [nosym_nofb_5c_iter5].
%:- [nosym_nofb_5c_iter6].

%:- [sym_prompt_verifier_5c_iter1].
:- [sym_prompt_verifier_5c_iter1_v1].
%:- [sym_prompt_verifier_5c_iter2].


% Helper function to check more than one examples at once.
test :-
    findall(E, example(E), Examples),
    check_all(Examples, 0, 0).

% Base case: no more examples -> print totals
check_all([], Pos, Neg) :-
    format('~nSummary:~n', []),
    format('  Positives (true): ~w~n', [Pos]),
    format('  Negatives (false): ~w~n', [Neg]).

% Recursive case: test each example
check_all([E|Rest], Pos, Neg) :-
    (   won_for_white(E)
    ->  format('won_for_white((~w)): true~n', [E]),
        Pos1 is Pos + 1,
        Neg1 = Neg
    ;   format('won_for_white((~w)): false~n', [E]),
        Neg1 is Neg + 1,
        Pos1 = Pos
    ),
    check_all(Rest, Pos1, Neg1).

