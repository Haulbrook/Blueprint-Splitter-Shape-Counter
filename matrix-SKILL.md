---
name: matrix
description: Master orchestrator that unifies all pipeline bots into one entity. Three modes - pitch (quick mockup), blueprint (customized spec), full (complete build). Use when you want the full bot pipeline or specific stages. Allows skipping stages if you already have intermediate outputs.
---

# MATRIX

The central nervous system. Connects all bots into a unified pipeline with multiple execution modes.

```
┌───────────────────────────────────────────────────────────────────────┐
│                              MATRIX                                   │
│                                                                       │
│  ┌─────────┐   ┌─────────┐   ┌───────────┐   ┌─────────────┐         │
│  │  idea   │ → │ prompt  │ → │ agentskill│ → │ bot-steroids│         │
│  │         │   │ booster │   │  builder  │   │  BASELINE   │         │
│  └─────────┘   └─────────┘   └───────────┘   └─────────────┘         │
│       │             │              │               │                  │
│       ▼             ▼              ▼               ▼                  │
│   [PITCH]      [PITCH]       [BLUEPRINT]     [BLUEPRINT]             │
│                                                    │                  │
│                     ┌─────────┐   ┌─────────┐     │                  │
│                     │guidance │ → │ call-to │ ←───┘                  │
│                     │counselor│   │ action  │                        │
│                     └─────────┘   └─────────┘                        │
│                          │             │                              │
│                          ▼             ▼                              │
│                     [BLUEPRINT]   [BLUEPRINT]                        │
│                                        │                              │
│       ┌─────────┐   ┌─────────────┐   │                              │
│       │  chaos  │ → │ bot-steroids│ ←─┘                              │
│       │ tester  │   │ REMEDIATION │                                  │
│       └─────────┘   └─────────────┘                                  │
│            │               │                                          │
│            ▼               ▼                                          │
│         [FULL]          [FULL]                                       │
│                            │                                          │
│                     ┌──────┴──────┐                                  │
│                     │   polish    │                                  │
│                     │   buffer    │                                  │
│                     │   stamp     │                                  │
│                     └─────────────┘                                  │
│                            │                                          │
│                            ▼                                          │
│                        [FULL] ✅                                      │
└───────────────────────────────────────────────────────────────────────┘
```

## Complete Bot Roster (9 Stages)

| # | Bot | Role | Source |
|---|-----|------|--------|
| 1 | **idea** | Generate and expand concepts | Built |
| 2 | **prompt-booster** | Enhance prompts with principles | Built |
| 3 | **agentskill-builder** | Create complete agents/skills | Your skill |
| 4 | **bot-steroids-baseline** | Initial training of agent | Your skill |
| 5 | **guidance-counselor** | Strategic planning + routing | Built |
| 6 | **call-to-action** | Executable steps + time estimates | Built |
| 7 | **chaos-tester** | Adversarial QA testing | Your skill |
| 8 | **bot-steroids-remediation** | Fix issues from testing | Your skill |
| 9 | **polish-buffer-stamp** | Final quality gate | Built |

## Two-Pass Training Strategy

**BASELINE (Stage 4)**: Train the agent immediately after creation
- Assesses initial fitness across 8 muscle groups
- Strengthens weak areas BEFORE planning begins
- Ensures all downstream work uses a "boosted" agent

**REMEDIATION (Stage 8)**: Fix issues found by chaos-tester
- Targets specific failures from adversarial testing
- More intense, focused training on broken areas
- Ensures production-ready quality

## Modes

### Mode 1: PITCH
Quick mockup for customer presentation.

```
INPUT:  Raw idea or concept
RUNS:   idea → prompt-booster
OUTPUT: Polished pitch template, presentation-ready concept
USE:    First client meeting, initial proposal, quick validation
```

### Mode 2: BLUEPRINT
Customized specification after client approval.

```
INPUT:  Confirmed concept (or raw idea)
RUNS:   idea → prompt-booster → agentskill-builder → bot-steroids-baseline → guidance-counselor → call-to-action
OUTPUT: Detailed spec with TRAINED agent, action steps, time estimates
USE:    Statement of work, project kickoff, scope agreement
```

### Mode 3: FULL
Complete top-to-bottom build with two training passes.

```
INPUT:  Approved blueprint (or raw idea)
RUNS:   idea → prompt-booster → agentskill-builder → bot-steroids-baseline → guidance-counselor → call-to-action → chaos-tester → bot-steroids-remediation → polish-buffer-stamp
OUTPUT: Twice-trained, tested, stamped deliverable - production-ready
USE:    Final delivery, production release, complete handoff
```

## Bot Responsibilities in Pipeline

| Stage | Bot | What It Does |
|-------|-----|--------------|
| 1 | idea | Expands raw concept into structured idea with alternatives |
| 2 | prompt-booster | Enhances with principles, clarification chains |
| 3 | agentskill-builder | Creates complete agent/skill structure if needed |
| 4 | bot-steroids-baseline | Initial training - boosts agent before planning |
| 5 | guidance-counselor | Strategic planning, decomposition, routing |
| 6 | call-to-action | Converts plan to numbered steps with code snippets |
| 7 | chaos-tester | Adversarial testing - tries to break everything |
| 8 | bot-steroids-remediation | Targeted training to fix test failures |
| 9 | polish-buffer-stamp | Final cleanup, verification, certification |

## Early Stage Expansion (Mode 3)

Mode 3 emphasizes **wide at top, narrow at execution**:

```
IDEA PHASE (expanded):
├── Generate 3-5 alternative approaches
├── Identify edge cases upfront
├── Map dependencies early
├── Flag potential blockers
└── Document assumptions

PROMPT PHASE (thorough):
├── Apply all relevant principles
├── Create clarification chains
├── Define acceptance criteria
├── Specify output formats
└── Include negative examples

GUIDANCE PHASE (detailed):
├── Micro-decomposition (smallest possible steps)
├── Dependency graph
├── Risk assessment per subtask
├── Parallel vs sequential identification
└── Strategic counsel for each subtask

EXECUTION PHASE (precise):
├── One action per step
├── Copy-paste code for each
├── Exact time estimates
├── Verification criteria
└── Rollback instructions
```

## Input Schema

```json
{
  "mode": "pitch|blueprint|full",
  "input": {
    "type": "raw|idea|prompt|plan|steps",
    "content": "Your idea or intermediate output here"
  },
  "options": {
    "skip_to": null,
    "detail_level": "normal|high|maximum",
    "include_alternatives": true,
    "customer_name": "Optional client name"
  }
}
```

## Output Schema

### Pitch Output
```json
{
  "mode": "pitch",
  "deliverable": {
    "title": "Project Title",
    "elevator_pitch": "One paragraph summary",
    "key_features": ["...", "..."],
    "value_proposition": "Why this matters",
    "rough_estimate": "Timeline range",
    "next_steps": "What we need to proceed"
  },
  "raw_outputs": {
    "idea": {...},
    "enhanced_prompt": {...}
  }
}
```

### Blueprint Output
```json
{
  "mode": "blueprint",
  "deliverable": {
    "title": "Project Title",
    "scope": {
      "included": ["...", "..."],
      "excluded": ["...", "..."],
      "assumptions": ["...", "..."]
    },
    "action_plan": {
      "phases": [...],
      "total_estimate": "X-Y hours/days",
      "milestones": [...]
    },
    "sign_off_required": true
  },
  "raw_outputs": {
    "idea": {...},
    "enhanced_prompt": {...},
    "strategic_plan": {...},
    "action_steps": {...}
  }
}
```

### Full Output
```json
{
  "mode": "full",
  "deliverable": {
    "manifest": {...},
    "files": [...],
    "stamp": {
      "id": "PBS-XXXXXXXX-XXXX",
      "seal": "✅ CERTIFIED",
      "score": 95
    }
  },
  "raw_outputs": {
    "idea": {...},
    "enhanced_prompt": {...},
    "strategic_plan": {...},
    "action_steps": {...},
    "test_results": {...},
    "polish_report": {...}
  },
  "pipeline_complete": true
}
```

## Skip-To Functionality

Already have intermediate output? Skip ahead:

```json
{
  "mode": "full",
  "input": {
    "type": "plan",
    "content": {... guidance-counselor output ...}
  },
  "options": {
    "skip_to": "call-to-action"
  }
}
```

Valid skip points:
- `prompt-booster` - Have idea, skip idea generation
- `guidance-counselor` - Have enhanced prompt, skip enhancement
- `call-to-action` - Have strategic plan, skip planning
- `testing-bot` - Have action steps, skip to testing
- `polish-buffer-stamp` - Have tested output, skip to final gate

## Confirmation Gates

Matrix pauses for confirmation at key points:

| Mode | Gate | What's Confirmed |
|------|------|------------------|
| pitch | End | "Ready to present?" |
| blueprint | After call-to-action | "Scope approved?" |
| full | After call-to-action | "Proceed to build?" |
| full | After testing | "Ready to stamp?" |

## Script Usage

Run full pipeline:
```bash
python scripts/orchestrate.py --mode full --input idea.json
```

Quick pitch:
```bash
python scripts/orchestrate.py --mode pitch --input idea.json
```

Skip to specific stage:
```bash
python scripts/orchestrate.py --mode blueprint --input prompt.json --skip-to guidance-counselor
```

## Integration Points

Matrix calls these bots in sequence:

| Bot | Import | Function |
|-----|--------|----------|
| idea | `idea/scripts/generate.py` | `generate_ideas()` |
| prompt-booster | `prompt-booster/scripts/enhance.py` | `enhance_prompt()` |
| guidance-counselor | `guidance-counselor/scripts/strategize.py` | `strategize()` |
| call-to-action | `call-to-action/scripts/generate_steps.py` | `generate_steps()` |
| testing-bot | `testing-bot/scripts/test.py` | `run_tests()` |
| polish-buffer-stamp | `polish-buffer-stamp/scripts/finalize.py` | `finalize()` |

## Error Handling

If any stage fails:
1. Log failure point
2. Save intermediate outputs
3. Report which bot failed
4. Allow resume from last successful stage

```json
{
  "error": {
    "stage": "guidance-counselor",
    "message": "Decomposition failed",
    "recovery": "Resume with: --skip-to guidance-counselor"
  },
  "partial_outputs": {
    "idea": {...},
    "enhanced_prompt": {...}
  }
}
```
