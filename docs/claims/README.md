# The book, claim by claim

López de Prado, *Advances in Financial Machine Learning* (Wiley, 2018), chapters 1 to 22, broken into 1,805 claims.
Each claim is one row in a chapter file, written in our own words and cited by section number. None quotes the
book, and no formula is reproduced: a row names the equation, snippet or figure it rests on (for example "Snippet
7.1" or "Figure 13.1") and says what the code does with it. The book is copyrighted and is not in this repository.

## How to read a chapter file

`chNN.md` holds one table: `id | section | kind | statement | where | evidence | status | note`.

| column | meaning |
|---|---|
| id | `I<chapter>.<n>`, stable, unique within the chapter |
| section | the book's heading number. Four-level headings are written `2.3.2(1)`, since a leak scanner reads `x.x.x.x` as an IP address |
| kind | logic, math, structure, algo, pitfall or constraint |
| statement | what the book asks or warns, paraphrased |
| where | the code that implements it, as `pmlab...` names |
| evidence | the test that settles it, as `tests/<file>.py::<test>`, and sometimes a study report |
| status | done (a named test), deferred (a reason and a date) or not applicable (a reason) |
| note | how the test settles it, or why the claim does not apply to a 5-minute binary market on one machine |

Counts by kind: logic 575, math 378, algo 248, structure 224, pitfall 217, constraint 163
(`python scripts/claims.py` prints the status totals; [the evidence page](../evidence.md) has the per-chapter test outcomes).

Some rows name files that stay in the private build repository: study reports (`research/*.md`), study scripts
(`scripts/*.py` other than the ones here), the first, per-heading pass (`docs/afml_sections/`) and a few design pages.
A claim whose only evidence is one of those is counted as "reference only" on the evidence page, never as passing.

## The second reading

Every chapter was read a second time by an agent that had not written it. It re-read each heading, counted the
claims the text makes, named any that had no row, and opened every test a "done" row cites to judge whether that test
could fail if the claim were false. The verdicts are in `audit/chNN.md`, one row per heading: 276 headings, 111 right
as written, 165 fixed; 1,826 claims counted in the text against 1,805 rows after the fixes.

What it kept finding, most frequent first:

- **Tests that could not fail.** Claims gated on a docstring matching a string ([audit/ch02](audit/ch02.md), 2.3.1(1));
  all six boosting steps of chapter 6 resting on one test that would still pass with the reweighting reversed, the
  discard rule removed or the vote unweighted ([audit/ch06](audit/ch06.md)).
- **Claims with no row at all**, among them snippets and figures that no row named ([audit/ch02](audit/ch02.md)).
- **Wrong reasons for deferring.** The three-state label of chapter 3 had no test and was deferred behind a code
  freeze for a pre-registered test; the freeze forbade changing that code, not testing it ([audit/ch03](audit/ch03.md)).

## Chapters

| file | chapter | file | chapter |
|---|---|---|---|
| [ch01](ch01.md) | Financial ML as a distinct subject | [ch12](ch12.md) | Backtesting through cross-validation |
| [ch02](ch02.md) | Financial data structures | [ch13](ch13.md) | Backtesting on synthetic data |
| [ch03](ch03.md) | Labeling | [ch14](ch14.md) | Backtest statistics |
| [ch04](ch04.md) | Sample weights | [ch15](ch15.md) | Understanding strategy risk |
| [ch05](ch05.md) | Fractionally differentiated features | [ch16](ch16.md) | Machine learning asset allocation |
| [ch06](ch06.md) | Ensemble methods | [ch17](ch17.md) | Structural breaks |
| [ch07](ch07.md) | Cross-validation in finance | [ch18](ch18.md) | Entropy features |
| [ch08](ch08.md) | Feature importance | [ch19](ch19.md) | Microstructural features |
| [ch09](ch09.md) | Hyper-parameter tuning with CV | [ch20](ch20.md) | Multiprocessing and vectorization |
| [ch10](ch10.md) | Bet sizing | [ch21](ch21.md) | Brute force and quantum computers |
| [ch11](ch11.md) | The dangers of backtesting | [ch22](ch22.md) | High-performance computational intelligence |
