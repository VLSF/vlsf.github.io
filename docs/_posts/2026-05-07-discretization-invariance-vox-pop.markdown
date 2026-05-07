---
layout: main
title:  "Discretization invariance for the people"
date:   2026-05-07
---

# Discretization invariance for the people

I wrote a blog post on discretization invariance — [blog](https://iclr-blogposts.github.io/2026/blog/2026/discretisation-invariance/), [repository](https://github.com/VLSF/discretisation_invariance), [poster](https://github.com/VLSF/discretisation_invariance/blob/main/assets/poster.pdf) — for ICLR 2026 with the main intention of initiating a discussion on the purpose of this property. Do we need discretization invariance and why? The question is almost never discussed seriously, yet dozens of discretization-invariant architectures have been developed over the years.

The definition can be found in the blog, the poster and—an especially simple and lucid one—in the article [Discretization Error of Fourier Neural Operators](https://arxiv.org/abs/2405.02221) by Samuel Lanthaler, Andrew M. Stuart and Margaret Trautner. We call an architecture discretization-invariant if it can be specified in terms of continuous operations: $$\inf$$, $$\sup$$, integrals, (partial) derivatives, [compositions](https://en.wikipedia.org/wiki/Composition_operator), etc. Sure, if you do not use any discretization, your architecture will be discretization invariant!

In the blog, I gathered all the arguments for and against discretization invariance I could locate in the literature or concoct myself.

## Arguments in favour of discretization invariance
1. **The discretization-invariant setup is theoretically natural.** The use of PDEs forces one to consider function spaces; thus, from a theoretical standpoint, it is natural to set up the learning problem in a Banach space.
2. **Multiscale training.** Discretization invariance is a form of "weight-sharing across different input resolutions"; for many architectures, one can use downsampled data to pre-train a model and later fine-tune it with high-resolution data.
3. **Reduced inference cost.** In applications where a model must be evaluated repeatedly (e.g., sampling from diffusion models, guidance, MCMC, or energy-based models), one can resort to a coarser model most of the time.
4. **Hyperparameter optimisation.** Hyperparameter optimisation can be performed with the help of a crude model, which can later be retrained on data with the desired resolution.

## Arguments against discretization invariance
1. **Discretization invariance is an asymptotic condition.** Formally, discretization invariance is only strictly achieved when the number of grid points tends to infinity.
2. **Only finite-resolution data is available.** Training, evaluation, and deployment are always performed in a fixed-resolution setting. A good practical example is weather forecasting based on the ERA5 dataset.
3. **No generalisation is observed when the grid is refined.** There is no "zero-shot super-resolution," and one should not expect an architecture to generalise beyond the resolution seen during the training stage.
4. **Discretization invariance does not guarantee good performance.** Any architecture can be made discretization-invariant with a suitable choice of encoder (e.g., scalar product, sampling) and decoder (e.g., interpolation, finite series as in DeepONet). Yet, the performance of such architectures can differ vastly.

## Vox pop
At the conference, I asked people to vote for or against discretization invariance. Most people I talked to were not doing "AI for science," likely with zero experience in operator learning for PDEs. People familiar with "AI for PDE" research were hedging. A few of them even refused to vote or took time to think and never returned to the poster (no offense taken, this is a huge conference).

<img src="{{ site.baseurl }}/assets/2026-05-07-2026-05-07-discretization-invariance-vox-pop/poster_with_votes.jpg" width="900">

According to my small survey, discretization invariance is winning 11 to 7. We have reached a democratic consensus. Discretization invariance is officially the way to go.
