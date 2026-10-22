# Baselines

`pfa gate` compares each run's reports with these numbers. An agent keeps the right to run alone
only while every check holds:

- **back office**: zero wrong actions in the replay and under fault injection, automation within
  2 points of the baseline, crash test, race and frozen-worker tests passed, shadow runs with no effects;
- **inspection**: defect recall within 2 points, false calls per board no more than 0.1 above,
  the worst-ranked 20% still holding the defects;
- **parts**: no more wrong designs programmed than the baseline, automation within 2 points.

A change of model, prompt (skill), effort level or checks that moves any of these fails CI. To accept
a deliberate trade-off, update the baseline in the same pull request, where it gets reviewed.
