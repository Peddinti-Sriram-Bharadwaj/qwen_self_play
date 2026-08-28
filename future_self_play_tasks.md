# Future Self-Play Benchmarks & Hardest Sequential Tasks

The hardest self-play sequential-learning tasks in current LLM literature are **formal theorem proving, adversarial code/test co-evolution, long-context multi-document reasoning, and open-ended skill/tool-use evolution**—not ordinary question answering.

There is no universally accepted difficulty leaderboard across self-play papers, so the ranking below is based on the hardest regimes actually constructed and evaluated: sparse or binary rewards, tasks beyond the base model’s capability, adversarial adaptation, long horizons, and reliable external verification.

| Rank | Task family | Why it is unusually hard | Strong literature instantiation |
|---|---|---|---|
| 1 | **Formal theorem proving with generated subproblems** | Proofs require exact, compiler-verified solutions; sufficiently hard problems have almost no reward; the conjecturer must generate useful intermediate problems rather than merely difficult ones. | [Scaling Self-Play with Self-Guidance](https://www.alphaxiv.org/abs/2604.20209) |
| 2 | **Adversarial code generation and test generation** | A coder tries to pass tests while a tester searches for implementation-specific bugs. Both agents improve, so the reward distribution continually moves. | [Code-A1](https://www.alphaxiv.org/abs/2603.15611) |
| 3 | **Hard goal-directed coding curricula** | The teacher must transform an unsolved real coding problem into learnable lemma–lift sequences without losing the algorithmic motif. | [GASP](https://www.alphaxiv.org/abs/2603.15957) |
| 4 | **Long-context multi-document reasoning** | The model must retrieve distributed evidence, ignore distractors, perform multi-hop reasoning, and generate a semantically correct answer that may not match a reference string exactly. | [SPELL](https://www.alphaxiv.org/abs/2509.23863) |
| 5 | **Open-ended skill and tool-use evolution** | The proposer must generate valid executable tasks while the system discovers, refines, and retires reusable skills. Difficulty is multidimensional rather than a single scalar. | [Skill Self-Play](https://www.alphaxiv.org/abs/2607.22529) |
| 6 | **Adversarial language games** | Attacker and defender must reason over a multi-turn conversation with asymmetric information, deception, implicit inference, and strict game rules. | [SPAG](https://www.alphaxiv.org/abs/2404.10642) |

## 1. Formal theorem proving is the strongest candidate

The most demanding established setting is probably **self-play Lean theorem proving**.

In [Scaling Self-Play with Self-Guidance](https://www.alphaxiv.org/abs/2604.20209), the solver must produce proofs accepted by the Lean compiler, while a conjecturer generates intermediate problems related to an unsolved target. The authors construct a dataset of roughly 3,300 difficult formal-math problems and define a harder subset of 1,346 problems that the strongest RL baseline never solves during the main run. Their self-guided method eventually solves close to 10% of this otherwise inaccessible subset. 

This is especially suitable for studying continual learning because the task frontier is naturally sequential:

```text
easy lemma → intermediate theorem → harder theorem → target theorem
```

The difficulty is not just solving a fixed task. The teacher must keep generating useful problems as the solver improves, while avoiding:
- impossible problems with zero reward;
- trivial problems with no learning signal;
- malformed or false theorems;
- synthetic tasks unrelated to the target;
- conjecturer collapse;
- solver entropy collapse.

**Best version for your project:** use a fixed set of hard Lean theorems and measure whether the model can progress through a sequence of automatically generated intermediate lemmas. This gives you both a self-play task and a clean sequential-learning benchmark.

## 2. Adversarial code/test co-evolution is probably the best practical task

If you need a task that is easier to implement than Lean but still genuinely difficult, use **adversarial software engineering**.

In [Code-A1](https://www.alphaxiv.org/abs/2603.15611), the Code LLM generates programs and the Test LLM inspects those programs to construct tests that expose subtle bugs. The coder is rewarded for passing the tests; the tester is rewarded for finding defects. Unlike ordinary self-play, the two roles are separated into different policies to avoid self-collusion.

The difficulty increases naturally:

```text
generic tests
→ edge-case tests
→ implementation-specific tests
→ adversarial tests targeting newly fixed bugs
```

The system maintains a “Mistake Book” of historical failures, so the coder must not only solve new tests but retain robustness against old failure cases. This is directly relevant to your plasticity/forgetting work: the Mistake Book creates an explicit sequential retention problem.

For a particularly hard benchmark, use:
- long-horizon repository-level code repair;
- hidden tests;
- mutation testing;
- adversarial generated tests;
- regression tests retained across training;
- multiple programming languages;
- resource limits and timeouts;
- bugs that require understanding interactions between functions.

This is likely the **best first task** for a small research project because correctness is executable, rewards are objective, and difficulty can be increased automatically.

## 3. Hard goalpost coding is a strong sequential curriculum

[GASP](https://www.alphaxiv.org/abs/2603.15957) provides a useful design for continual self-play: begin with a real coding problem that the base model cannot solve, then generate a sequence of easier and harder variants.

Its pipeline is:
```text
unsolved real problem
        ↓
learnable lemma
        ↓
harder lift
        ↓
repeated curriculum
        ↓
original goalpost
```

This is a good task for testing whether a model loses plasticity because you can measure:
- whether it learns each new lift;
- whether it retains earlier lemmas;
- whether it can relearn failed goalposts;
- whether curriculum difficulty continues to increase;
- whether the teacher collapses to trivial or malformed tasks.

## 4. Long-context reasoning is hard in a different way

[SPELL](https://www.alphaxiv.org/abs/2509.23863) uses a questioner, responder, and verifier to create self-play over long documents. The questioner generates questions from a subset of documents; the responder must answer using the full document set, including distractors; and the verifier checks semantic equivalence.

The difficulty grows as the questioner’s history memory expands:
```text
single-document question
→ multi-document question
→ distributed evidence
→ distractor-heavy multi-hop question
→ long-context numerical or relational reasoning
```

## 5. Open-ended skill and tool-use evolution tests broader continual learning

[Skill Self-Play](https://www.alphaxiv.org/abs/2607.22529) tests a more realistic form of continual self-improvement. The model must learn executable tool-use behaviors and logical reasoning while the system evolves a skill library.

For your purposes, the hardest version would combine:
1. a tool-use task;
2. a changing API schema;
3. multi-step tool calls;
4. hidden state;
5. irreversible actions;
6. constraints that must remain satisfied across the entire episode.

That would let you test both **plasticity** and **catastrophic forgetting**:
- new tools and schemas test plasticity;
- old tools and schemas test retention;
- changing action conventions tests interference;
- a persistent skill library tests whether learned procedures remain accessible.

## My recommendation for your project

If your goal is specifically **continual self-play plus plasticity diagnostics**, I would use three tiers:

### Tier 1: executable coding curriculum
Start with GASP-style hard coding tasks:
- select real problems with `pass@100 = 0`;
- generate lemma–lift sequences;
- retain hidden tests;
- measure old-task retention and new-task adaptation;
- add regression tests as a memory buffer.

### Tier 2: adversarial tester–coder co-evolution
Then add Code-A1-style adversarial testing:
- the tester sees the candidate code;
- the tester generates targeted hidden tests;
- old failures are retained;
- the coder must solve increasingly adversarial tests;
- the tester must discover new bugs after previous bugs are fixed.

### Tier 3: Lean theorem proving
Use formal theorem proving for the hardest, most publication-worthy extension.
The central experiment could be:
> Does a solver trained sequentially on self-generated intermediate lemmas retain the ability to learn new proof patterns, or does it eventually become unable to acquire novel proof strategies?

**If you want one task to begin with tomorrow, choose adversarial code/test co-evolution. If you want the hardest long-term research direction, choose Lean theorem proving with guided intermediate-lemma self-play.**
