### Podsumowanie wykonanego eksperymentu

- Użyto 20000 przykładów treningowych, 5000 walidacyjnych i 10000 testowych; wykonano 5 epok na wariant.
- Accuracy testowa: **A = 96.43%, B = 90.19%, C = 24.95%**.
- Ranking w tym uruchomieniu: **A > B > C** (ewentualne remisy rozstrzyga kolejność wariantów).
- Różnica A minus B: +6.24 punktów procentowych; A minus C: +71.48 pp.
- Równomierne zgadywanie daje średnio 10%; przewidywanie najczęstszej klasy testowej daje 11.35% (opisowy punkt odniesienia).
- Wybrano epoki na podstawie straty walidacyjnej, bez dostrajania do accuracy testowej.

Na wykresach t-SNE sprawdź, w których warstwach punkty o tym samym kolorze tworzą lokalne grupy. W opisie raportu odwołaj się do konkretnych paneli, a jako miarę skuteczności podaj accuracy i macierze pomyłek. Wyniki dotyczą pojedynczego ziarna i ograniczonego czasu uczenia.
