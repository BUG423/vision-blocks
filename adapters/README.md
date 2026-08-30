# Input adapters

Adapters convert non-image inputs into explicit tensor contracts for the reusable modules in this repository.

## BCL

`bcl/` contains one-dimensional modules and the BCL input adapter formerly maintained on the separate `bcl` branch. The expected input layout is `[batch, channels, time]`. Individual files include a minimal executable example.

These implementations are experimental. Validate data semantics, masking behavior, normalization, and downstream accuracy before production or publication use.
