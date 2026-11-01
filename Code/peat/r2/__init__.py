"""
PEAT revision-2 experiments (APIN-D-26-06244R1, Reviewers #4 and #5).

Every experiment is a self-contained *cell* (train and/or evaluate one model
variant for one seed) so that cells can run on any worker VM and be resumed
independently. Results land in results/r2/cells/<cell_id>/ and are pushed to
GitHub after each cell. See Code/run_r2.py for the single entry point.
"""
