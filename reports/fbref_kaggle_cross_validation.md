# FBref (soccerdata) vs Kaggle — 2024-25 cross-validation

xG is excluded (soccerdata's FBref tables don't provide it). Players matched by normalized name within league.

## Premier League

- FBref players: 562 | Kaggle players: 562 | matched: 558 (99.3% of FBref)

| stat | Pearson r | MAE | mean FBref | mean Kaggle |
|---|---|---|---|---|
| goals | 1.000 | 0.00 | 1.94 | 1.94 |
| assists | 1.000 | 0.00 | 1.44 | 1.44 |
| minutes | 1.000 | 0.00 | 1341.49 | 1341.49 |
| shots | 1.000 | 0.17 | 17.61 | 17.47 |

## La Liga

- FBref players: 588 | Kaggle players: 588 | matched: 581 (98.8% of FBref)

| stat | Pearson r | MAE | mean FBref | mean Kaggle |
|---|---|---|---|---|
| goals | 1.000 | 0.00 | 1.66 | 1.66 |
| assists | 1.000 | 0.00 | 1.16 | 1.16 |
| minutes | 1.000 | 0.00 | 1282.41 | 1282.41 |
| shots | 0.999 | 0.24 | 15.48 | 15.27 |
