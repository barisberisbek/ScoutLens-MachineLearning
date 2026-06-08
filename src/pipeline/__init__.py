"""Inference pipeline: Stage 1 → Feature Forwarder → Stage 2.

The Feature Forwarder is a deterministic transform (not ML): it ages a player's
current-season feature row forward one season (age+1, contract−12mo, lag shift, projected
stats swapped in) so the Stage-2 valuation model produces a forward-looking value.
"""
