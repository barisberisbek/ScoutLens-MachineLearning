# Transfer Fees Cross-Validation (2024-25 top-5)

- Source A: custom Transfermarkt scrape (this phase)
- Source B: davidcariboo `tm_transfers.parquet` (Phase 1C, fee>0)

## Counts
- Source A total arrivals: 1757
- Source A fee>0: 548
- Source B fee>0: 121
- Name overlap (A∩B): 63
- Source A max fee: €80,000,000 | >=€20M: 108 | >=€60M: 9

## Big transfers (>=€20M) found in scrape, ABSENT from davidcariboo
- Khvicha Kvaratskhelia: €80,000,000 (Napoli → Paris Saint-Germain, Ligue 1)
- Julián Álvarez: €75,000,000 (Man City → Atlético de Madrid, La Liga)
- Omar Marmoush: €75,000,000 (Frankfurt → Manchester City, Premier League)
- João Neves: €65,920,000 (Benfica → Paris Saint-Germain, Ligue 1)
- Dominic Solanke: €64,300,000 (Bournemouth → Tottenham Hotspur, Premier League)
- Leny Yoro: €62,000,000 (Lille → Manchester United, Premier League)
- Dani Olmo: €61,000,000 (Leipzig → FC Barcelona, La Liga)
- Nico González: €60,000,000 (Porto → Manchester City, Premier League)
- Pedro Neto: €60,000,000 (Wolves → Chelsea FC, Premier League)
- Amadou Onana: €59,350,000 (Everton → Aston Villa, Premier League)