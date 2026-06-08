# Name Resolution Audit (Phase 2)

FBref backbone resolution decisions: **20557** rows.

## Resolution method distribution

| method | count | % |
|---|---|---|
| exact | 14743 | 71.7% |
| unmatched | 3644 | 17.7% |
| fuzzy_nat | 1150 | 5.6% |
| exact_name_year | 964 | 4.7% |
| fuzzy | 56 | 0.3% |

**TM-matched: 16857 (82.0%)** · synthetic/unmatched: 3700 (18.0%)

## TM-match rate by league

| league | rows | matched % |
|---|---|---|
| Belgian Pro League | 2002 | 78.1% |
| Bundesliga | 1908 | 84.5% |
| Eredivisie | 2018 | 84.0% |
| La Liga | 2311 | 84.0% |
| Liga Portugal | 2199 | 71.8% |
| Ligue 1 | 2127 | 82.7% |
| Premier League | 2126 | 86.5% |
| Serie A | 2282 | 83.3% |
| Süper Lig | 2383 | 80.2% |

## Understat orphans (matched to FBref backbone)

Unmatched Understat player-rows: **269**. These are logged, never silently dropped; an orphan rate above 10% would signal an FBref coverage gap.

| player | team | season |
|---|---|---|
| Wesley | Aston Villa | 2021-22 |
| Zanka | Brentford | 2021-22 |
| Bryan Gil Salvatierra | Tottenham | 2021-22 |
| Pierre-Emile Højbjerg | Tottenham | 2021-22 |
| Tanguy NDombele Alvaro | Tottenham | 2021-22 |
| Juan Camilo Hernández | Watford | 2021-22 |
| Adama Traoré | Wolverhampton Wanderers | 2021-22 |
| Chiquinho | Wolverhampton Wanderers | 2021-22 |
| Hee-Chan Hwang | Wolverhampton Wanderers | 2021-22 |
| Jonny | Wolverhampton Wanderers | 2021-22 |
| Toti | Wolverhampton Wanderers | 2021-22 |
| Trincão | Wolverhampton Wanderers | 2021-22 |
| Zanka | Brentford | 2022-23 |
| Victor Kristiansen | Leicester | 2022-23 |
| Arnaut Danjuma Groeneveld | Tottenham | 2022-23 |
| Bryan Gil Salvatierra | Tottenham | 2022-23 |
| Pape Sarr | Tottenham | 2022-23 |
| Pierre-Emile Højbjerg | Tottenham | 2022-23 |
| Hee-Chan Hwang | Wolverhampton Wanderers | 2022-23 |
| Jonny | Wolverhampton Wanderers | 2022-23 |
| João Moutinho | Wolverhampton Wanderers | 2022-23 |
| Toti | Wolverhampton Wanderers | 2022-23 |
| Louis Beyer | Burnley | 2023-24 |
| Chimuanya Ugochukwu | Chelsea | 2023-24 |
| Mads Andersen | Luton | 2023-24 |
| … | … | (+244 more) |

## 30 lowest-score accepted fuzzy matches (spot-check these)

| source | input_name | birth_year | nationality | score | player_id |
|---|---|---|---|---|---|
| fbref | Juan Carlos Familia | 2000.0 | DOM | 85.0 | 349567 |
| fbref | Valentino Livramento | 2002.0 | ENG | 85.71 | 503981 |
| fbref | Emerson | 1999.0 | BRA | 85.71 | 607854 |
| fbref | Faustino Anjorin | 2001.0 | ENG | 85.71 | 433181 |
| fbref | Abdoulaye Seck | 1992.0 | SEN | 85.71 | 193584 |
| fbref | Valentino Livramento | 2002.0 | ENG | 85.71 | 503981 |
| fbref | Emerson | 1999.0 | BRA | 85.71 | 607854 |
| fbref | Emerson | 1999.0 | BRA | 85.71 | 607854 |
| fbref | Valentino Livramento | 2002.0 | ENG | 85.71 | 503981 |
| fbref | Emerson | 1999.0 | BRA | 85.71 | 607854 |
| fbref | Emerson | 1999.0 | BRA | 85.71 | 607854 |
| fbref | Valentino Livramento | 2002.0 | ENG | 85.71 | 503981 |
| fbref | Xusniddin Aliqulov | 1999.0 | UZB | 86.49 | 581429 |
| fbref | Xusniddin Aliqulov | 1999.0 | UZB | 86.49 | 581429 |
| fbref | Yehor Yarmoliuk | 2004.0 | UKR | 86.67 | 717411 |
| fbref | Arsenii Batahov | 2002.0 | UKR | 86.67 | 665048 |
| fbref | Yehor Yarmoliuk | 2004.0 | UKR | 86.67 | 717411 |
| fbref | Ronaldo Tavares | 1997.0 | POR | 86.67 | 376731 |
| fbref | Omenuke Mfulu | 1994.0 | COD | 86.96 | 203516 |
| fbref | Babatunde Akinsola | 2003.0 | NGA | 87.5 | 924447 |
| fbref | Babatunde Akinsola | 2003.0 | NGA | 87.5 | 924447 |
| fbref | Dire Mebude | 2004.0 | SCO | 88.0 | 719672 |
| fbref | Dire Mebude | 2004.0 | SCO | 88.0 | 719672 |
| fbref | Abdenasser El Khayati | 1989.0 | NED | 89.47 | 88968 |
| fbref | Dapo Afolayan | 1997.0 | ENG | 89.66 | 487850 |
| fbref | Cemalil Sertel | 2000.0 | TUR | 90.0 | 573981 |
| fbref | Joshua Acheampong | 2006.0 | ENG | 90.0 | 1004708 |
| fbref | Illia Zabarnyi | 2002.0 | UKR | 90.0 | 659089 |
| fbref | Emi Buendía | 1996.0 | ARG | 90.0 | 321247 |
| fbref | Marquinhos | 2003.0 | BRA | 90.0 | 887834 |
