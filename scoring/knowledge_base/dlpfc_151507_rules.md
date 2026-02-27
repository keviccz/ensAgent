## DLPFC_151507 Annotation Critic Rules (Minimal KB)

This KB is used by the **Critic Agent** to validate Annotator outputs.

### Allowed Labels
`Layer 1, Layer 2, Layer 3, Layer 4, Layer 5, Layer 6, White Matter, Mixed L6/White Matter, Mixed L1/L2, Mixed L2/L3, Mixed L3/L4, Mixed L4/L5, Mixed L5/L6, Mixed, Unknown`

### High-level constraints
- The final `biological_identity` must be exactly one of the allowed labels.
- Reasoning must cite evidence from the provided expert outputs (Marker/Pathway/Spatiality/Visual).
- Do not accept over-confident claims when evidence is mixed; prefer `Mixed` or `Unknown`.

### White Matter trap (blocker)
- **Do NOT accept `White Matter`** if myelin/oligodendrocyte evidence is not present.
- Axonal genes (e.g., **NEFL/NEFM**) are **not sufficient** for White Matter.
- White Matter requires strong oligodendrocyte/myelin markers such as **MBP, PLP1, MOBP, MOG, MAG, CNP**.
- **WM-adjacent deep boundary**: if only limited myelin markers are present (e.g., **MBP+PLP1**) and Visual/Spatiality morphology looks like a cortical band (Layer 5/6), prefer **`Mixed L6/White Matter`** over pure `White Matter`.

### Superficial glia-rich trap (common confusion)
- A superficial band may show **GFAP/AQP4/ALDH1L1** (astrocytic) and be peripheral/edge-like in spatial prior.
- If neuron markers are absent and GFAP is high, do not label as deep layers or white matter.

### Granular/Layer 4 heuristic (non-blocker, use cautiously)
- Layer 4 may show strong neuronal activity signatures; can include **RORB** and synaptic/oxidative pathways.
- DLPFC is not always strongly granular; if ambiguous between L3/L4, allow `Mixed L2/L3` or `Layer 3`/`Layer 4` with low confidence and clear alternatives.

### Layer marker priors (grounded gene rules)
- **Layer 1**: **RELN**, **CXCL14**, **CPLX3**
- **Layer 2/3 (IT)**: **CUX1**, **CUX2**, **CALB1**, **SATB2**
- **Layer 4 (IT)**: **RORB** (strong), **PCP4** (supportive)
- **Layer 5**: **BCL11B (CTIP2)**, **FEZF2**, **TSHZ2**
- **Layer 6**: **TLE4**, **FOXP2**, **CRYM**
- **White Matter**: multiple oligodendrocyte/myelin genes (**MBP**, **PLP1**, **MOG**, **MAG**, **MOBP**, **CNP**, **CLDN11**) + oligodendrocyte enrichment

### Non-specific neuron genes (do not use alone to decide layer)
- **NEFL/NEFM**, **MAP1B**, **TUBA1B** and other generic axonal/cytoskeletal genes are not layer-specific.

### Visual required (blocker)
- If Visual expert output is missing, failed, or flagged as low quality, the run must fail and request Visual rerun.
- Visual must focus on domain morphology (contiguity, fragmentation, band-like structure) and must not score color/contrast aesthetics.




