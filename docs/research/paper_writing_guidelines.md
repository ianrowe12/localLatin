# ACL / ARR Paper Writing Guidelines

## 1. Research Goal

A strong paper should **advance understanding of an important problem**, not just improve metrics.

Focus on:

* Important problem
* Novel idea or insight
* Convincing experiments
* Clear writing

Bad papers usually:

* Only tweak existing methods
* Report better numbers without insight
* Solve narrow or artificial problems

Good papers **teach the community something new**.

---

# 2. Core Paper Narrative

Every paper must clearly answer four questions:

1. **What problem are we solving?**
2. **Why is it important?**
3. **What is the key idea?**
4. **Why does the method work?**

Typical structure:

```
Title
Abstract
1. Introduction
2. Related Work
3. Method
4. Experiments
5. Analysis / Discussion
6. Conclusion
```

The **introduction is the most important section**.

It should:

* motivate the problem
* summarize the key idea
* clearly state contributions

---

# 3. Contributions

Clearly list contributions at the end of the introduction.

Typical types:

* **Methodological**: new algorithm or method
* **Empirical**: new results or benchmark
* **Scientific insight**: understanding of model behavior

Contributions should be **crisp and specific**.

Bad:

> We improve performance.

Good:

> We propose a monolingual method for identifying language-specific SAE features.

---

# 4. Writing Style

Prioritize **clarity over elegance**.

Rules:

* Use short sentences
* Avoid complex grammar
* Avoid unnecessary jargon
* Introduce one idea per sentence
* Prefer active voice

Example:

Bad:

```
It should be noted that the following method is capable of producing...
```

Good:

```
Our method produces...
```

Readers should understand the paper even when **skimming**.

---

# 5. Organizing Ideas

Structure ideas hierarchically.

Each section should follow:

```
Overview → Key idea → Technical details → Summary
```

Guidelines:

* Start sections with a short overview paragraph
* Clearly separate new ideas from implementation details
* Motivate every technical step

---

# 6. Related Work

Do not list papers individually.

Instead:

* group prior work into categories
* explain how they relate to your approach
* clarify what gap your work fills

Avoid hostile language.

Bad:

```
Previous work fails because...
```

Better:

```
Previous work focuses on X, while we address Y.
```

---

# 7. Figures

Figures are critical.

Many readers browse papers through figures.

Good figures:

* explain the method visually
* have self-contained captions
* clearly label all components

Important figure types:

* **Money figure** (method overview)
* Results comparison
* Ablation visualization

Captions should allow understanding **without reading the text**.

---

# 8. Experiments

Experiments must **support the paper's claims**.

Include:

* evaluation setup
* datasets
* baselines
* metrics
* implementation details

Results should:

* compare against strong baselines
* include ablations
* demonstrate why the method works

Avoid reporting numbers without interpretation.

Explain:

* why improvements occur
* when the method fails

---

# 9. Reproducibility

Papers should allow others to reproduce results.

Include:

* datasets
* model checkpoints
* hyperparameters
* training setup
* evaluation protocol

Use appendix for detailed implementation information.

---

# 10. Paper Planning Workflow

Write the paper iteratively.

Recommended workflow:

```
1. Title
2. Paper outline
3. Introduction outline
4. Introduction draft
5. Related work
6. Method
7. Experiment design
8. Results
9. Abstract
10. Conclusion
```

Write **skeletons first**, then expand.

---

# 11. Revision Process

Writing is iterative.

Steps:

1. Write
2. Rewrite
3. Simplify
4. Remove unnecessary text
5. Repeat

Good editing rule:

> Every sentence must serve a purpose.

---

# 12. Common Paper Problems

Common reasons papers get rejected:

* unclear contribution
* weak experimental validation
* poor writing
* incremental novelty
* unclear motivation

Bad writing alone can cause rejection even with good ideas.

---

# 13. Final Paper Checklist

Before submission, ensure a reader can:

* identify the problem quickly
* understand the main idea
* recognize what is new
* follow the method
* reproduce the experiments
* understand why the work matters

If any of these fail, revise the paper.
