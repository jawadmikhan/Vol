# Systematic Volatility Portfolio Process

Interactive workflow dashboard for a five-leg systematic volatility portfolio — from signal construction through execution, monitoring, and rebalancing.

**Live site:** [jawadmikhan.github.io/Vol](https://jawadmikhan.github.io/Vol)  
**Workflow Map:** [jawadmikhan.github.io/Vol/workflow.html](https://jawadmikhan.github.io/Vol/workflow.html)

---

## Overview

A 48-step, 7-stage portfolio management workflow spanning five strategy legs:

| Leg | Strategy | Description |
|-----|----------|-------------|
| 1 | Dispersion (Core) | Index vs constituent volatility mispricing |
| 2 | Volatility Harvesting | Systematic short volatility premium capture |
| 3 | Directional Long/Short | Macro-regime driven volatility views |
| 4 | Dynamic Volatility Targeting | Inverse-volatility leverage scaling |
| 5 | Option Overlays | Protective puts, collars, tail-risk hedges |

---

## Workflow Stages

| Stage | Name | Steps |
|-------|------|-------|
| 1 | Idea Generation & Sourcing | 7 |
| 2 | Research Aggregation | 6 |
| 3 | Collaboration & Peer Review | 5 |
| 4 | Thesis Development | 11 |
| 5 | Decision, Approval, & Portfolio Construction | 7 |
| 6 | Monitoring & Review | 6 |
| 7 | Rebalancing & Liquidity | 6 |

---

## Blocker Distribution

| Category | Steps | Share |
|----------|-------|-------|
| No Blocker (Green) | 34 | 71% |
| Soft Blocker (Yellow) | 10 | 21% |
| Hard Blocker (Red) | 4 | 8% |

**Soft Blockers** require simulated or pre-fetched data (e.g., live volatility surfaces, VIX data feeds).  
**Hard Blockers** require physical presence or proprietary system access (e.g., exchange execution, prime broker reconciliation).

---

## Gate Architecture

| Gate | Location | Decisions |
|------|----------|-----------|
| Signal Quality Review | Step 3.1 | Approve / Modify / Reject |
| Investment Committee Approval | Step 5.2 | Approved / Conditional / Rejected |
| Compliance Clearance | Step 5.3 | Clear / Fail |
| Thesis Drift Assessment | Step 6.4 | Maintain / Adjust / Unwind |
| Action Decision | After Step 7.6 | Rebalance / Hold / Full Rebuild |

---

## Feedback Loops

| Type | Path | Description |
|------|------|-------------|
| Tactical | 7.4 → 6.1 | Post-execution back to daily monitoring |
| Strategic | 6.3 → 1.5 | Regime shift triggers signal review |
| Fundamental | 6.4 → 1.1 | Thesis unwind triggers full universe rebuild |

---

## Task Streams

Task streams are the longest uninterrupted chains of No Blocker and Soft Blocker steps — the foundation for prompt creation.

| Stream | Range | Steps | Terminates At |
|--------|-------|-------|---------------|
| A — Primary | 1.1 → 5.1 | 30 | 5.2 Investment Committee Vote (Hard) |
| B | 5.7 → 7.3 | 10 | 7.4 Trade Execution (Hard) |
| C | 5.3 → 5.4 | 2 | 5.5 Exchange Execution (Hard) |
| D | 7.5 → 7.6 | 2 | End of cycle |

---

## Pages

### `index.html` — Interactive Dashboard
- Dark theme with tabbed navigation across all 7 stages
- Mermaid flowcharts with actor-prefixed nodes on every tab
- Master flow diagram covering the full end-to-end process
- Step detail tables with goal, actor, blocker, inputs, and outputs
- Reference tables, color legend, task stream cards, and strategy leg mapping

### `workflow.html` — Standardized Workflow Map
- White background, print-friendly layout
- 3-color blocker scheme with standard flowchart shapes (oval, rectangle, diamond, parallelogram)
- Full 48-step diagram with all feedback loops and gate decisions
- Back to Dashboard navigation link

---

## Structure

```
Vol/
├── index.html       # Interactive dashboard (dark theme)
├── workflow.html    # Standardized workflow map (light theme)
└── README.md
```

Self-contained HTML files — no build step, no external dependencies beyond Mermaid (loaded via CDN).
