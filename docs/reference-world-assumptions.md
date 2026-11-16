# Reference world assumptions

The reference world is a simulated county court that acts as "reality" for the experiments.
Every setting in `config/reference_world.yaml` is either **sourced** from official figures or
an **assumption**. Hidden relationships are deliberately not given to generators, so the
evaluation can test whether they were inferred.

Snapshot date: 30 June 2026. Issue dates: 1 January 2024 to 30 June 2026. The working dataset has 2,000 fictional cases; the "Simulated result" column comes from a 200,000-case check run with the same settings (small samples are noisier, especially trial timings).

## Sources

- [Civil Justice Statistics Quarterly: April to June 2026](https://www.gov.uk/government/statistics/civil-justice-statistics-quarterly-april-to-june-2026/civil-justice-statistics-quarterly-april-to-june-2026) (Ministry of Justice, published 3 September 2026, provisional). Open Government Licence v3.0.
- [Civil Procedure Rules, rule 26.9](https://www.legislation.gov.uk/uksi/1998/3132/rule/26.9/2024-04-06/data.html) (version dated 6 April 2024). Open Government Licence v3.0. Check for later amendments before finalising.

## Settings

| Setting | Value | Status | Reason or source | Simulated result (200,000-case check) |
| --- | --- | --- | --- | --- |
| Claim mix | Money 87%, personal injury 2%, other damages 2.5%, possession 5%, other non-money 3.5% | Sourced | Claims received, Q2 2026: 571,000 total; money 495,000; personal injury 11,000; other damages 15,000; possession 29,000; other non-money about 20,000 | Within 0.1 percentage points |
| Money claims under £500 | About 50% | Sourced | 245,000 of 495,000 money claims valued under £500 | 46% (all claims with a value) |
| Money claim values | Two groups: median £420 (spread 1.2) and 9% higher-value with median £13,000 (spread 0.45) | Assumption | Shapes chosen so value bands and track allocation match published shares | See below |
| Small claims limit | £10,000 (personal injury £1,500) | Sourced | CPR 26.9(1) and (4); road traffic personal injury limits simplified to the general £1,500 | All small claims at or under the limit |
| Fast track limit | £25,000 | Sourced | CPR 26.9(5)(b)(i) | |
| Intermediate track limit | £100,000 | Sourced | CPR 26.9(7)(b) | |
| Claims over £25,000 sent to multi-track | 70% | Assumption | Tuned so intermediate and multi-track shares are close to published | Intermediate 1.3%, multi 3.2% (published 2%, 5%) |
| Defended share | 14% of claims | Sourced | 82,000 defended, 14% of claims issued | 14.0% |
| Defence more likely for higher values | Logistic on log value | Assumption (hidden) | Higher-value claims are more often worth contesting | Under £500: 4.8%; £10k to £25k: 49%; over £100k: 88% |
| Track allocation | Small 66%, fast 26%, intermediate 2%, multi 5% | Sourced | Allocations Q2 2026 | Small 68%, fast 27%, intermediate 1.3%, multi 3.2% |
| Representation, defended claims | Both 37%, claimant only 41%, defendant only 3%, neither 19% | Sourced | Defended claims by representation | Within 0.5 percentage points |
| Representation, undefended claims | Claimant 60%, defendant 2% | Assumption | Bulk creditors usually use solicitors; undefended defendants rarely do | |
| Interpreter needed | 4% | Assumption | No published figure | 4.0% |
| Defended claims reaching trial | About 15% | Sourced ratio | 12,000 trials / 82,000 defences | 15.2% |
| Litigants in person settle less | Higher trial chance if neither side represented; lower if both | Assumption (hidden) | Represented parties more often negotiate settlement | |
| Small claims share of trials | 75% | Sourced | 8,900 of 12,000 trials | 68% |
| Median issue to trial, small claims | 41.0 weeks | Sourced | Q2 2026; base median 280 days, tuned so the observed median matches | 41.0 weeks |
| Median issue to trial, other tracks | 55.9 weeks | Sourced | Q2 2026; base median 382 days, tuned so the observed median matches | 55.8 weeks |
| Court speed differences | Each court has a hidden speed factor (geometric mean 1, own random stream) | Assumption (hidden) | Real courts differ in workload | Visible in held-back statistics by court |
| Slower when neither side represented, or interpreter needed | 10% and 8% longer | Assumption (hidden) | Extra case management time | 47.9 weeks when neither side represented, 44.0 otherwise |
| Claimant wins at trial | 62% | Assumption | No published figure | |
| Default judgments | About 94% of judgments | Sourced | Q2 2026 | 93.5% of money judgments |
| Judgments by admission | About 4% | Sourced | 15,000 judgments by admission | 4.9% of all judgments |
| Days from issue to default judgment | 19 to 90 | Assumption | After the response period | |
| Undefended claims withdrawn | 10% | Assumption | Paid or discontinued | |
| Warrants per judgment | About 20% | Sourced ratio | 73,000 warrants / 368,000 judgments | 21.0% |
| Low-value claims enforced more | Higher warrant chance under £1,000; lower if defendant represented | Assumption (hidden) | Small debts are often enforced by warrant of control | 23% under £1,000; 17% at £1,000 and over |
| Possession hearings | Median 70 days after issue; 85% order; 30% warrant | Assumption | No figure used from the bulletin | |
| Courts | 12 public county court hearing centre names with regions | Assumption | Even spread; real volumes differ | |

## Known simplifications

- Volumes are not modelled; only proportions and timings are calibrated.
- The road traffic personal injury small claims limit (£5,000, or £1,000 in some cases) is simplified to £1,500.
- Possession claims are not allocated to tracks.
- Trials are single events; adjournments and multi-day hearings are not modelled.
- Published figures are quarterly and provisional; the world uses them as steady-state averages over 2024 to 2026.

## Review

| Reviewed by | Role | Date | Comments |
| --- | --- | --- | --- |
| | Domain expert (for example a listing officer or civil practitioner) | | |
