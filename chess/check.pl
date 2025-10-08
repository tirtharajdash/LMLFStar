:- [krk].

test :-
    findall(E, example(E), Examples),
    partition(won_for_white, Examples, Positives, Negatives),
    print_results(Positives, Negatives).

print_results(Positives, Negatives) :-
    forall(member(E, Positives),
           format('won_for_white((~q)): true~n', [E])),
    forall(member(E, Negatives),
           format('won_for_white((~q)): false~n', [E])),
    nl,
    length(Positives, Pcount),
    length(Negatives, Ncount),
    format('Summary:~n', []),
    format('  Positives (true): ~w~n', [Pcount]),
    format('  Negatives (false): ~w~n', [Ncount]).

