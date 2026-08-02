---
layout: main
title:  "Deep brachistochrone: part 2"
date:   2026-08-03
asset_path: "/assets/2026-08-03-brachistochrone-part-2/"
references:
  - key: "gelfand_eng"
    author: "Gelfand, Israel M., Fomin, Sergei V."
    title: "Calculus of Variations."
    year: 2000
    note: "Translation of a classical book (Вариационное исчисление, Гельфанд, Фомин, 1961) made by Richard A. Silverman. I used this book to justify the global optimality of brachistochrone. Local optimality (weak extremum) is discussed in Chapter 5. Global optimality (strong extremum) is explained in Chapter 6."

  - key: "gelfand_ru"
    author: "Гельфанд, Израиль Моисеевич, Фомин, Сергей Васильевич"
    title: "Вариационное исчисление."
    year: 1961

  - key: "kahan1980"
    author: "Kahan, William M."
    year: 1980
    title: "Handheld Calculator Evaluates Integrals."
    url: "https://hparchive.com/Journals/HPJ-1980-08.pdf"
    note: "The link is on the whole August issue of HPJ, 1980. Be aware of the dangers around you."

  - key: "backward2013"
    author: "Fillion, Nicolas. Corless, Robert M."
    year: 2013
    title: "A Graduate Introduction to Numerical Methods: From the Viewpoint of Backward Error Analysis."
    url: "https://link.springer.com/book/10.1007/978-1-4614-8453-0"
    note: "An outstanding textbook on numerical analysis. Here, in Chapter 10, I first learned about the Kahan's argument on the impossibility of integration."

  - key: "yarotsky2013"
    author: "Yarotsky, Dmitry."
    year: 2016
    title: "Error bounds for approximations with deep ReLU networks."
    url: "https://arxiv.org/abs/1610.01145"

  - key: "deepquads2021"
    author: "Rivera, Jon A. Taylor, Jamie M. Omella, Ángel J., Pardo, David."
    year: 2021
    title: "On quadrature rules for solving partial differential equations using neural networks."
    url: "https://arxiv.org/abs/2111.00217"
    note: "A very nice article that discusses numerical troubles one get into when trying to solve variational problems with deep learning. The discussion is quite different from what I have here, but some overlap is certainly present. For example, most remedies I consider are also covered in this article."

  - key: "OL2022"
    author: "Prach, Bernd. Lampert, Christoph H."
    year: 2022
    title: "Almost-Orthogonal Layers for Efficient General-Purpose Lipschitz Networks."
    url: "https://arxiv.org/abs/2208.03160"
    note: "A numerical linear algebra way to 1-Lipschitz network."
---

# Deep brachistochrone: part 2

TL;DR: We first see that the Brachistochrone solution obtained in [Deep brachistochrone]({% post_url 2026-06-06-brachistochrone-part-1 %}) is globally optimal: no other curve can have a shorter travel time! Next, we discuss what went wrong with our neural network and optimization that produced a much better travel time. Integration is to blame; we see why and provide defense mechanisms.

## Optimality of cycloid

In [Deep brachistochrone]({% post_url 2026-06-06-brachistochrone-part-1 %}), we found the cycloid as a solution to the Euler–Lagrange equations. These equations are only a necessary condition for an extremum.

Consider a finite-dimensional optimization problem as an analogy:

$$
x^{\star} = \arg \min_{y} f(y): y, x^{\star} \in \mathbb{R}^{n}, f(y) \in \mathbb{R}.
$$

Finding a solution to the Euler–Lagrange equations is equivalent to solving $\frac{d f(x)}{dx} = 0$. To ensure that we have a local minimum, we also need to check that $\frac{d^2 f(x)}{dx^2} > 0$ at $x^{\star}$. For a global minimum, the global properties of $f(x)$ must be studied, which is typically much harder. The most familiar and arguably simplest case where we can claim a global minimum has been found is when $f(x)$ is strongly convex. In this case, [a unique global minimum exists](https://en.wikipedia.org/wiki/Convex_function#Properties_of_strongly-convex_functions). Can we do something similar for variational problems? Absolutely.

For variational problems, local optimality is quite easy to establish. To do that, we study the second variation and prove that it is positive on the solution curve.

<details>
  <summary><b>Local optimality of cycloid</b></summary>

  <div style="margin-top: 10px; padding-left: 15px; border-left: 2px solid #ccc;" markdown="1">

  The material here is based on Chapter 5 of ([Gelfand, Fomin, 2000](#gelfand_eng)).

  For local optimality, we need to study the second variation:

  $$
  \begin{split}
  \int F\left(f + h, f^{'} + h^{'}\right) du &= \int F\left(f, f^{'}\right) du + \underbrace{\int \left(\frac{\partial F\left(f, f^{'}\right)}{\partial f} h(u) + \frac{\partial F\left(f, f^{'}\right)}{\partial f^{'}} h^{'}(u)\right) du}_{\text{first variation}}\\
  &+\underbrace{\frac{1}{2}\int \left(\frac{\partial^2 F\left(f, f^{'}\right)}{\partial f^2} \left(h(u)\right)^2 + 2\frac{\partial^2 F\left(f, f^{'}\right)}{\partial f \partial f^{'}} h(u)h^{'}(u) + \frac{\partial^2 F\left(f, f^{'}\right)}{\partial \left(f^{'}\right)^{2}} \left(h^{'}(u)\right)^2\right) du}_{\text{second variation}} + \dots
  \end{split}
  $$

  By assumption, we already solved the Euler-Lagrange equation, so the first variation is zero for an arbitrary $h(u)$ compatible with the boundary conditions. To show local optimality, we need to check two conditions on the second variation.

  The first (necessary) condition, due to Legendre, is $F_{f'f'} \geq 0$. To see why this is necessary, first rearrange the terms:

  $$
  \begin{split}
  &\int \left(\frac{\partial^2 F\left(f, f^{'}\right)}{\partial f^2} \left(h(u)\right)^2 + 2\frac{\partial^2 F\left(f, f^{'}\right)}{\partial f \partial f^{'}} h(u)h^{'}(u) + \frac{\partial^2 F\left(f, f^{'}\right)}{\partial \left(f^{'}\right)^{2}} \left(h^{'}(u)\right)^2\right) du\\
  &= \int \left(\left(\frac{\partial^2 F\left(f, f^{'}\right)}{\partial f^2} - \frac{d}{du} \left(\frac{\partial^2 F\left(f, f^{'}\right)}{\partial f \partial f^{'}}\right) \right) \left(h(u)\right)^2 + \frac{\partial^2 F\left(f, f^{'}\right)}{\partial \left(f^{'}\right)^{2}} \left(h^{'}(u)\right)^2\right) du.
  \end{split}
  $$

  If $F_{f'f'} < 0$ somewhere, we can locally select $h(u)$ such that $\|h(u)\|$ is arbitrarily small, and $\|h'(u)\|$ is arbitrarily large. A simple example is $h(u) = \epsilon\sin(u\big/\epsilon^2) \underset{\epsilon\rightarrow 0}{\longrightarrow} 0$ with derivative $h'(u) = \cos(u\big/\epsilon^2)\big/\epsilon \underset{\epsilon\rightarrow 0}{\longrightarrow} \infty$.

  The second (sufficient) condition is that the second variation can be transformed into a perfect square. More specifically, if we rewrite the second variation as $\delta^2 J = \int \left(P \left(h'\right)^2 + Q \left(h\right)^2\right) du$, the conditions are:
  + $P > 0$
  + $-\frac{d}{du}\left(P y'\right) + Q y = 0$ has a solution $y(u)$ that does not vanish anywhere on the domain of $u$ (no conjugate points).

  If these conditions hold, we can indeed safely transform the second variation into a perfect square. First, we observe that for an arbitrary continuously differentiable $w$, we have:

  $$
    \int_{a}^{b} \frac{d}{du}\left(w h ^2\right) du = \left.w h ^2\right|_{a}^{b} = 0.
  $$

  Next, we write:

  $$
    \delta^2 J = \int \left(P \left(h'\right)^2 + 2 w h h^{'} + \left(Q + w^{'}\right) \left(h\right)^2\right) du = \int P\left(h^{'} + \alpha \frac{h}{\sqrt{P}}\right)^2 du,
  $$

  where the last equality holds for $\alpha = w\big/ \sqrt{P}$ with $\alpha^2 = Q + w'$, or equivalently, $P(Q + w') = w^2$. If we take $w = - y' P \big/ y$ for some $y$, the equivalent condition on $w$ reads $-\frac{d}{du}\left(P y'\right) + Q y = 0$. If this equation has a non-vanishing solution, $w$ is well-defined and the second variation is a perfect square.

  Also observe that the expression in brackets $h' + \alpha \frac{h}{\sqrt{P}}$ cannot vanish unless $h(u) = 0$. This is the case because $h(0) = 0$ and the said expression, viewed as an initial-value problem, has a unique solution.

  Now we are ready to check these two conditions for our cycloid solution:

  $$
    \begin{split}
      &f(v) = 1 + \cos(v),\\
      &u(v) = v + \sin(v) + \pi,
    \end{split}
  $$

  for the functional:

  $$
  F\left(v, f, f^{'}\right) = \frac{\sqrt{1 + \left(\frac{f^{'}}{1 + \cos v}\right)^2}}{\sqrt{f}} (1  + \cos v).
  $$

  After some algebraic manipulations, we get:

  $$
  \begin{split}
  &F_{f^{'}} = \frac{f^{'}}{\sqrt{f}\sqrt{1 + \left(\frac{f^{'}}{1 + \cos v}\right)^2}}\frac{1}{1 + \cos v},\\
  &F_{f^{'}f^{'}} = \frac{1}{\sqrt{f}\left(1 + \left(\frac{f^{'}}{1 + \cos v}\right)^2\right)^{3/2}}\frac{1}{1 + \cos v},\\
  &F_{f^{'}f} = -\frac{1}{2}\frac{f^{'}}{f^{3/2}\sqrt{1 + \left(\frac{f^{'}}{1 + \cos v}\right)^2}}\frac{1}{1 + \cos v},\\
  &F_{ff} = \frac{3}{4}\frac{\sqrt{1 + \left(\frac{f^{'}}{1 + \cos v}\right)^2}}{f^{5/2}} (1  + \cos v).
  \end{split}
  $$

  It is also easy to see that:

  $$
  1 + \left(\frac{f^{'}}{1 + \cos v}\right)^2 = \frac{2}{1 + \cos v}
  $$

  and from that:

  $$
  \begin{split}
  &F_{f^{'}f^{'}} = 2^{-3/2},\\
  &F_{f^{'}f} = 2^{-3/2} \frac{\sin v}{(1 + \cos v)^2},\\
  &F_{ff} = 2^{-3/2} \frac{3}{(1 + \cos v)^2}.
  \end{split}
  $$

  The Legendre condition is obviously satisfied. We first find the second variation:

  $$
  \left(\frac{\partial^2 F\left(f, f^{'}\right)}{\partial f^2} - \frac{d}{dv} \left(\frac{\partial^2 F\left(f, f^{'}\right)}{\partial f \partial f^{'}}\right) \right) = 2^{-3/2} \frac{1}{1 + \cos v} \Rightarrow \delta^2 J = \frac{1}{2^{5/2}}\int_{-\pi}^{0}\left(\frac{h^2}{1 + \cos v} + \left(h^{'}\right)^2\right) dv.
  $$

  Since both terms are positive, and $\delta^2 J = 0$ iff $h(v) = 0, h'(v) = 0$, the second variation is clearly positive. We also check the equation from the sufficient conditions:

  $$
  -\frac{d^2 y}{dv^2} + \frac{1}{1 + \cos v} y = 0 \Rightarrow y(v) = c_1 \tan\left(\frac{v}{2}\right) + c_2 \left(2 + v \tan\left(\frac{v}{2}\right)\right).
  $$

  For $v \in (-\pi, 0)$, the function $2 + v \tan(v/2)$ is strictly positive, so we can take $c_1 = 0$ and $c_2 = 1$ to obtain a valid, non-vanishing solution.
  </div>
</details>

Global optimality is much harder to study, but luckily for us, our functional is convex.

<details>
  <summary><b>Global optimality of cycloid</b></summary>

  <div style="margin-top: 10px; padding-left: 15px; border-left: 2px solid #ccc;" markdown="1">

  Detailed proofs can be found in Chapter 6 of ([Gelfand, Fomin, 2000](#gelfand_eng)).

  Consider the Lagrangian $F(v, f, f')$ and the functional $J[f] = \int F(v, f(v), f'(v)) dv$. If we have a candidate for a global minimum $f_0$, it can be compared with an arbitrary curve $w$ with the help of the Weierstrass excess functional:

  $$
  J[w] - J[f_0] = \int E(v, w(v), p(v, w(v)), w^{'}(v)) dv,
  $$

  where the integrand is known as the Weierstrass excess function (or $E$-function):

  $$
  E(v, u, p, q) = F(v, u, q) - F(v, u, p) - (q - p) F_{u^{'}}(v, u, p),
  $$

  and $p(v, w(v))$ is the slope of an extremal from a field of extremals.

  A field of extremals is an interesting geometric object, a portion of which you can see below.

  <figure style="text-align: center;">
      <img src="{{ page.asset_path }}field_of_extremals.svg" width="700" alt="Field of extremals diagram">
      <figcaption style="font-style: italic; color: #555; margin-top: 8px;">
        Artist's view of the field of extremals. Each black curve corresponds to a unique extremal passing through the black point and the origin. The red curve is the target extremal, a candidate for the global optimum. The space around the red curve is packed with other extremals, such that a unique extremal passes through each point.
      </figcaption>
  </figure>

  It is built by embedding a potential global minimum $f_0$ into other extremals (i.e., solutions of the Euler-Lagrange equations) in such a way that only one extremal passes through each point $(v, f)$. When such a field is available, we define $p(v, f)$ to be the slope of the extremal that passes through $(v, f)$ at this particular point. This slope is what we have above in the Weierstrass excess function. The field of extremals explicitly depends on $f_0$, and because of that, the function $E$ also depends on $f_0$, which may not be obvious from the chosen notation.

  If we can show that $\int E(v, w(v), p(v, w(v)), w^{'}(v)) dv > 0$ for an arbitrary admissible $w$, we can be sure the solution is globally optimal. It might look like we need to build this field of extremals, but in our case, there is a shortcut.

  To use it, we first observe that $E$ is just a function of four real variables, so we can use Taylor's theorem with the Lagrange remainder to transform it into:

  $$
  E(v, u, p, q) = F(v, u, q) - F(v, u, p) - (q - p) F_{u^{'}}(v, u, p) = \frac{1}{2}(q - p)^2 F_{u^{'} u^{'}}(v, u, \xi),\,\xi\in[p, q].
  $$

  We already found that

  $$
  F_{f^{'}f^{'}}(v, f, f^{'}) = \frac{1}{\sqrt{f}\left(1 + \left(\frac{f^{'}}{1 + \cos v}\right)^2\right)^{3/2}}\frac{1}{1 + \cos v} = \frac{(1 + \cos v)^2}{\sqrt{f}\left((1 + \cos v)^2 + \left(f^{'}\right)^2\right)^{3/2}}.
  $$

  Clearly $F_{f^{'}f^{'}}(v, a, b) > 0$ for every $a > 0$ and arbitrary $b$, so the integral of the Weierstrass excess function is positive unless $w$ and $f_0$ have the same slope at each point. In this case $w = f_0$, which finishes our proof (by intimidation) of global optimality.
  </div>
</details>

If the cycloid is optimal and has travel time $\sqrt{2g} t = 4.443$, how come our neural network has time $\sqrt{2g} t = 2.221$?

## Post-mortem

Our problem is geometric. To identify what's wrong, let's look at the deep learning solution and compare it with the cycloid and other curves. Maybe we can spot something suspicious in the pictures.

<img src="{{ page.asset_path }}solutions.svg" width="800">

Well, the neural network is definitely thinking outside the box. We can see that the bead is initially in free fall, and after that, it climbs upward to the destination point. The computed time for such a trajectory is $\sqrt{2g} t = 2.221$. Can we trust it?

Recall that we are discussing the curve connecting two points $(0, 0)$ and $(\pi, -2)$ that results in the best possible travel time for a bead attached to it. The gravity force is pointing downward as usual, and we assume a force $mg$. If we imagine that gravity acts instead along the straight line between these two points, the travel time would be $\sqrt{\pi^2 + 4} = \frac{gt^2}{2}$, which gives us $\sqrt{2g} t = 2 \left(\pi^2 + 4\right)^{1/4} \simeq 3.712$. Yet, the curve produced by the neural network results in a shorter travel time, despite the fact that the bead is forced to move upward in part of the trajectory. The way we compute time must be incorrect.

The analytic expression for the travel time from [Deep brachistochrone]({% post_url 2026-06-06-brachistochrone-part-1 %}) reads

$$
\int_{0}^{x_1}\frac{\sqrt{1 + \left(f^{'}(u)\right)^2}du}{\sqrt{f(u)}} = \sqrt{2g} t.
$$

To compute this expression for a given curve $f(u)$, we must know how to: (i) perform elementary operations (division, taking squares, and square roots), (ii) compute the derivative, and (iii) compute the integral.

The first operation is available to us out of the box from JAX, and even [IEEE 754](https://en.wikipedia.org/wiki/IEEE_754) floating-point standard implementations are advised to contain such basic functions. Looking at the graph, I see no unusual values, so a failure of (i) is unlikely.

We also compute derivatives, but this is again available out of the box in JAX because it implements [automatic differentiation](https://docs.jax.dev/en/latest/automatic-differentiation.html). AD certainly can fail, and we seem to have some discontinuity in our neural network, so a failure of (ii) is not completely out of the question.

Still, our main suspect is the failure of integration. The reason is that we cannot integrate without assumptions (more about that later), and the assumptions we make can become unreasonable when deep learning is applied. Since we used [Gauss quadrature](https://en.wikipedia.org/wiki/Gaussian_quadrature), we assumed that the function is well-captured by Legendre polynomials. A simple test is to evaluate time on several distinct grids and see whether the time we report remains stable.

To obtain the data, the network and the cubic model are trained from scratch on the grid with 128 points, and after that, the travel time for both models is computed on several grids of different sizes.

||neural network|cubic|
|$N$|$\sqrt{2g}t$|$\sqrt{2g}t$|
|:---:|:---:|:---:|
|$32$|$5.448$|$4.425$|
|$64$|$5.065$|$4.463$|
|$\underline{128}$|$\underline{2.287}$|$\underline{4.483}$|
|$256$|$5.999$|$4.492$|
|$512$|$5.587$|$4.497$|
|$1024$|$5.507$|$4.500$|
{: style="--table-padding: 20px;"}

Our suspicion just got confirmed. Somehow, our integration, which works reasonably well for the cubic model, completely fails for the neural network: when we step outside the computation grid used at training time, our integral estimation falls apart. The instability is so profound that the travel time changes by a factor of 3 if we change the grid size by a mere 10 points.

<img src="{{ page.asset_path }}travel_time_quadrature.svg" width="800">

What is wrong with integration? To get the answer, we first need to go back to 1980 and ask [William Kahan](https://en.wikipedia.org/wiki/William_Kahan), a hero of numerical analysis.

## William Kahan's impossibility of deterministic integration

In 1980, William Kahan published a paper in the Hewlett-Packard Journal ([Kahan, 1980](#kahan1980)). In this article, Kahan summarized how he was involved in the implementation of integration on a handheld calculator, the HP-34C. The article contains lots of interesting insights on integration, practical numerical analysis, and also some fun reports on the time needed for certain integrals. The only result that we need, though, can be neatly summarized by paraphrasing a great textbook ([Fillion, Corless, 1980](#backward2013)):

**Kahan's theorem.** Deterministic numerical integration of functions defined by procedures ("black boxes") is impossible, even for analytic functions, unless the class of integrands represented by the black boxes is further constrained.
{: .theorem }

By assumption, a deterministic numerical integration evaluates a given function $f$ at a finite, predefined set of points $x_{1}, \dots, x_{K}$. In principle, the number of points $K$ and the points $x_k$ themselves can depend on the values of $f$ evaluated at all previous points $f(x_i), i < k$. The strategy is to locate these points first and then construct a special function that is integrated incorrectly when evaluated at these specific points.

First, we implement a function that always returns zero and records the query points:

```python
buffer = []

def spy(x):
    buffer.append(x)
    return 0
```

Then, this function is integrated with a given routine. For convenience, I will demonstrate the construction using the SciPy [quad](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.quad.html#scipy.integrate.quad) function, which is a wrapper around [QUADPACK](https://en.wikipedia.org/wiki/QUADPACK):

```python
from scipy.integrate import quad

a, b = 0, 1
_ = quad(spy, a, b)
```

As a result, we know the points $y_i, i=1,\dots,K$ where `quad` samples the function:

```python
print(*[f"{f:.3f}" for f in buffer])
> 0.500 0.013 0.987 0.067 0.933 0.160 0.840 0.283 0.717 0.426 0.574 0.002 0.998 0.035 0.965 0.110 0.890 0.219 0.781 0.353 0.647
```

It is clear that any function that returns zero at these specific points is indistinguishable from $f(x) = 0$ for all $x$. There are lots of such functions. In particular, we can select $g(x) = \psi(x)(x - y_1)^2 \cdots (x - y_{K})^2$ with an arbitrary $\psi(x)$.

```python
def adversarial_input(x, buffer, s):
    return s * np.prod((x - buffer)**2, axis=0)

s = 1e25
integral = quad(adversarial_input, a, b, args=(buffer, s))
print(integral)
> (0.0, 0.0)
```

For an example with $\psi(x) = \text{const}$, `quad` estimates the integral to be $0$ with high confidence ($0$ absolute error). Too bad the function is positive between the points $y_i, i=1,\dots, K$, so the integral cannot be zero, as is evident from the plot of $g(x)$:

<img src="{{ page.asset_path }}Kahan_adversarial.svg" width="450">

With Kahan's construction, we can propose functions that are integrated with arbitrarily bad absolute error. This is possible because function spaces are infinite-dimensional, while our data is finite. Given such a disadvantage, one should not expect much from worst-case performance.

## What is wrong with Deep Brachistochrone

Now we are ready to explain why deep learning failed on the brachistochrone problem. The argument combines three ideas: (i) error analysis for the integration problem, (ii) Kahan's result, and (iii) the excellent approximation properties of neural networks.

When we replace integration with a quadrature rule, we introduce error. Our integration rule is [Gauss-Legendre quadrature](https://en.wikipedia.org/wiki/Gauss%E2%80%93Legendre_quadrature), so it is convenient to represent an integrand as a series of Legendre polynomials $f(x) \approx \sum_{i=0}^{K} c_i p_i(x)$. With a large $K$, [we can approximate any continuous function arbitrarily well](https://en.wikipedia.org/wiki/Stone%E2%80%93Weierstrass_theorem). I denote the exact integral as $\mathcal{I}(f) := \int f(x) dx$ and the quadrature rule as $I_{N}(f) := \sum_{i=1}^{N} \omega_{i} f(x_i)$. Gauss quadrature of order $N$ is exact for all polynomials with order up to $2N - 1$, so we have

$$
\int f(x) dx \approx \underset{\text{integral approximation}}{\sum_{i=0}^{2N-1} c_i \mathcal{I}(p_i)} - \underset{\text{integration error}}{\sum_{i=2N}^{K} c_i \left(I(p_i) - \mathcal{I}(p_i)\right)}.
$$

Note that $\approx$ appears because the series is considered to give an approximate representation, i.e., $f(x) \approx \sum_{i=0}^{K} c_i p_i(x)$. The integration error is completely accounted for by the split into two terms. The same expression remains correct for our problem with $f(x)$ replaced by $F(x; v) := \sqrt{1 + \left(v'(x)\right)^2}\big/\sqrt{v(x)}$, which is just a function of $x$ for a given $v(x)$.

Since we are optimizing the integral approximation, our problem changes to

$$
\theta^{\star} = \arg \min_{\theta} \left(\int_{0}^{x_1}\frac{\sqrt{1 + \left(f_{\theta}^{'}(u)\right)^2}du}{\sqrt{f_{\theta}(u)}} + \sum_{i=2N}^{K} c_i(f_{\theta}, f_{\theta}^{'}) \left(I(p_i) - \mathcal{I}(p_i)\right)\right).
$$

The second expression can be quite hard to compute. To analyze it, we observe that the sum of two terms is a quadrature rule. By Kahan's argument, we can construct a function such that the whole expression in brackets is zero. This function would give zero travel time for the Brachistochrone problem. Of course, the argument holds only if we are entitled to select an arbitrary integrand. We can't, since our integrand is constrained.

Fortunately for us, neural networks are known to have extremely good approximation rates ([Yarotsky, 2013](#yarotsky2013)). Given that, it is plausible that the flexibility of neural networks can overcome the constraints imposed by a particular form of the functional.

In short, the argument is:
1. By using quadrature rules, we are optimizing the wrong functional.
2. Kahan's argument implies that for a modified functional, a zero solution exists.
3. Neural networks present a strong model with excellent approximation properties, so a new optimal travel time (zero) is likely within reach.


## Saving Deep Brachistochrone

What can we do to save the day? Our options, overlapping with the ones considered in ([Rivera, Taylor, Omella, Pardo, 2021](#deepquads2021)), are:
1. Use the strong form in place of the weak one.
2. Decrease the capacity of the model: constraining coefficients, decreasing the size, or simplifying the model.
3. Increase the accuracy of integration: a larger number of quadrature points, more integration rules, or using adaptive quadratures.
4. Make the quadrature rule non-deterministic: Monte Carlo methods.

### Strong form

The root of our problem is numerical quadrature. Why not get rid of integration altogether? To do that, we use a strong form of necessary optimality conditions, given by the Euler–Lagrange equations:

$$
\frac{d}{du}\left(\frac{f^{'}(u)}{\sqrt{f(u)\left(1 + \left(f^{'}(u)\right)^2\right)}}\right) = -\frac{1}{2}\frac{\sqrt{1 + \left(f^{'}(u)\right)^2}}{\left(f(u)\right)^{3/2}}, f(0) = 0, f(\pi) = 2.
$$

<details>
  <summary><b>Numerical solution with a strong form</b></summary>

  <div style="margin-top: 10px; padding-left: 15px; border-left: 2px solid #ccc;" markdown="1">

  ```python
  import jax.numpy as jnp
  import optax

  from scipy.special import roots_legendre
  from jax.lax import scan
  from jax.nn import gelu
  from jax import grad, value_and_grad, random, vmap

  def get_weights_and_grid(N, x0, x1, y0, y1):
      grid, weights = roots_legendre(N)
      grid = jnp.array((x1 - x0) * (grid + 1) / 2 + x0).reshape(-1, 1)
      weights = (x1 - x0) * jnp.array(weights) / 2
      return weights, grid

  def compute_time(model_params, grid, weights, model):
      dy = vmap(grad(lambda x: model(model_params, x)[0]))(grid)[:, 0]
      y = vmap(model, in_axes=(None, 0))(model_params, grid)[:, 0]
      S = jnp.sqrt(1 + dy**2) / jnp.sqrt(jnp.abs(y))
      res = jnp.sum(weights * S)
      return res

  def init_NN(key, N_features, N_layers):
      N_in, N_pr, N_out = N_features
      N_features_ = [N_in,] + [N_pr,]*N_layers + [N_out,]
      model = dict()
      model['b'] = [jnp.zeros((n_out,)) + 0.1 for n_in, n_out in zip(N_features_[:-1], N_features_[1:])]
      model['W'] = [2 * random.normal(key, (n_out, n_in)) / jnp.sqrt(n_out + n_in) for n_in, n_out, key in zip(N_features_[:-1], N_features_[1:], random.split(key, N_layers+1))]
      return model

  def NN(model_params, x, x0, x1, y0, y1):
      eps = 1e-3
      y = model_params['W'][0] @ x + model_params['b'][0]
      for w, b in zip(model_params['W'][1:], model_params['b'][1:]):
          y = gelu(y)
          y = w @ y + b
      y_ = y0 * (x - x1) / (x0 - x1) + y1 * (x - x0) / (x1 - x0) + eps * y * (x - x0) * (x - x1)
      return y_

  def get_lhs(model_params, x, model):
      dy = grad(lambda x: model(model_params, x)[0])(x)
      y = model(model_params, x)[0]
      lhs = dy / (jnp.sqrt(y*(1 + dy**2)))
      return lhs

  def strong_loss(model_params, grid, model):
      d_lhs = vmap(grad(lambda x: get_lhs(model_params, x, model)[0]))(grid)[:, 0]
      dy = vmap(grad(lambda x: model(model_params, x)[0]))(grid)[:, 0]
      y = vmap(model, in_axes=(None, 0))(model_params, grid)[:, 0]
      rhs = -0.5*jnp.sqrt(1 + dy**2) / jnp.sqrt(y**3)
      res = jnp.sum((d_lhs - rhs)**2)
      return res

  def make_step_scan(carry, ind, optim, model):
      model_params, opt_state, grid_x, grid, weights = carry
      l, grads = value_and_grad(lambda a, b: strong_loss(a, b, model))(model_params, grid_x)
      travel_time = compute_time(model_params, grid, weights, model)
      updates, opt_state = optim.update(grads, opt_state)
      model_params = optax.apply_updates(model_params, updates)
      return [model_params, opt_state, grid_x, grid, weights], (l, travel_time)

  N = 512
  x0, x1 = 0, jnp.pi
  y0, y1 = 0, 2
  fine_weights, fine_grid = get_weights_and_grid(N, x0, x1, y0, y1)

  N = 128
  x0, x1 = 0, jnp.pi
  y0, y1 = 0, 2
  grid_x = (jnp.linspace(x0, x1, N+2)[1:-1]).reshape(-1, 1)
  weights, grid = get_weights_and_grid(N, x0, x1, y0, y1)

  model_ = lambda model_params, x: NN(model_params, x, x0, x1, y0, y1)
  key = random.PRNGKey(33)
  N_features = [1, 100, 1]
  N_layers = 3
  model_params_ = init_NN(key, N_features, N_layers)

  learning_rate = 1e-4
  optim = optax.adam(learning_rate=learning_rate)
  opt_state = optim.init(model_params_)

  make_step_scan_ = lambda a, b: make_step_scan(a, b, optim, model_)

  N_updates = 50000
  ind = jnp.arange(N_updates)
  carry, (history, travel_time) = scan(make_step_scan_, [model_params_, opt_state, grid_x, grid, weights], ind)
  model_params_ = carry[0]
  ```

  </div>
</details>

Training is going fine, and travel time is stable. The solution slowly converges to the cycloid.

<img src="{{ page.asset_path }}strong_form_training.svg" width="800">

Travel time, estimated with Gauss-Legendre quadrature on the grid with $N$ points, also remains stable:

|$N$|$\sqrt{2g}t$|
|:---:|:---:|
|$8$|$3.914$|
|$16$|$4.252$|
|$32$|$4.346$|
|$64$|$4.410$|
|$128$|$4.441$|
|$256$|$4.455$|
|$512$|$4.463$|
|$1024$|$4.467$|
{: style="--table-padding: 20px;"}

A remnant of the quadrature is still present, because convergence is strongly influenced by the choice of the collocation points. If one replaces `grid_x` (a uniform grid) with `grid` (Legendre points), convergence to the true cycloid is not observed.

### Decreasing the model's capacity

We have already seen that simple quadratic and cubic models do not lead to pathological solutions. Similarly, a quadrature rule can handle smaller neural networks better. Below is an example of three learning curves for networks with an increasing number of layers:

<img src="{{ page.asset_path }}learning_curve_vs_capacity.svg" width="450">

The more parameters the network has, the faster it produces the unphysical solution. A little cherry-picking is involved in this result. In real experiments, one should expect lots of variability from non-convex (and usually stochastic) optimization.

There are other ways to decrease model capacity. The most notable one is regularization. For example, one can restrict the Lipschitz constant of a neural network by limiting the spectral radii of its weight matrices. If you ask me, I would recommend an almost orthogonal layer, as proposed in ([Prach, Lampert, 2022](#OL2022)).

### Better deterministic integration

It is clear from Kahan's construction that it will be harder and harder for a neural network to exploit the integration kernel once the accuracy of the quadrature rule increases. There are many ways to increase accuracy: [Romberg's method](https://en.wikipedia.org/wiki/Romberg%27s_method), local or global adaptive quadratures, quadratures of increased order, etc. As an example, I tried to increase the order of the Legendre-Gauss quadrature and recorded the results, which you can see below:

<img src="{{ page.asset_path }}learning_curve_vs_quadrature_order.svg" width="450">

Evidently, with a higher quadrature order, it becomes harder for a neural network to yield an unphysical time. For $N_x = 512$ and $N_x = 1024$, the stability is quite good. Alas, increasing the order does not completely suppress adversarial effects.

### Monte Carlo

A necessary condition for an adversarial attack on integration was the predictability of integration points. With [Monte Carlo integration](https://en.wikipedia.org/wiki/Monte_Carlo_integration), points become random. Recall that Monte Carlo integration is based on the identity for the mean of a uniformly distributed random variable $I = \int\_{a}^{b} f(x) dx = \mathbb{E}\_{x\sim \text{Ind}[a, b]}[f(x)]$, which is replaced by sampling in practical implementations: $I \approx \frac{(b-a)}{N}\sum\_{i=1}^{N} f(x_i),\,x\_i\sim\text{Ind}[a, b] \text{ i.i.d.}$

<details>
  <summary><b>Numerical solution with the Monte Carlo integration</b></summary>

  <div style="margin-top: 10px; padding-left: 15px; border-left: 2px solid #ccc;" markdown="1">

  ```python
  import jax.numpy as jnp
  import optax

  from scipy.special import roots_legendre
  from jax.lax import scan
  from jax.nn import gelu
  from jax import grad, value_and_grad, random, vmap

  def get_weights_and_grid(N, x0, x1, y0, y1):
      grid, weights = roots_legendre(N)
      grid = jnp.array((x1 - x0) * (grid + 1) / 2 + x0).reshape(-1, 1)
      weights = (x1 - x0) * jnp.array(weights) / 2
      return weights, grid

  def compute_time(model_params, grid, weights, model):
      dy = vmap(grad(lambda x: model(model_params, x)[0]))(grid)[:, 0]
      y = vmap(model, in_axes=(None, 0))(model_params, grid)[:, 0]
      S = jnp.sqrt(1 + dy**2) / jnp.sqrt(jnp.abs(y))
      res = jnp.sum(weights * S)
      return res

  def init_NN(key, N_features, N_layers):
      N_in, N_pr, N_out = N_features
      N_features_ = [N_in,] + [N_pr,]*N_layers + [N_out,]
      model = dict()
      model['b'] = [jnp.zeros((n_out,)) + 0.1 for n_in, n_out in zip(N_features_[:-1], N_features_[1:])]
      model['W'] = [2 * random.normal(key, (n_out, n_in)) / jnp.sqrt(n_out + n_in) for n_in, n_out, key in zip(N_features_[:-1], N_features_[1:], random.split(key, N_layers+1))]
      return model

  def NN(model_params, x, x0, x1, y0, y1):
      eps = 1e-3
      y = model_params['W'][0] @ x + model_params['b'][0]
      for w, b in zip(model_params['W'][1:], model_params['b'][1:]):
          y = gelu(y)
          y = w @ y + b
      y_ = y0 * (x - x1) / (x0 - x1) + y1 * (x - x0) / (x1 - x0) + eps * y * (x - x0) * (x - x1)
      return y_

  def get_lhs(model_params, x, model):
      dy = grad(lambda x: model(model_params, x)[0])(x)
      y = model(model_params, x)[0]
      lhs = dy / (jnp.sqrt(y*(1 + dy**2)))
      return lhs

  def strong_loss(model_params, grid, model):
      d_lhs = vmap(grad(lambda x: get_lhs(model_params, x, model)[0]))(grid)[:, 0]
      dy = vmap(grad(lambda x: model(model_params, x)[0]))(grid)[:, 0]
      y = vmap(model, in_axes=(None, 0))(model_params, grid)[:, 0]
      rhs = -0.5*jnp.sqrt(1 + dy**2) / jnp.sqrt(y**3)
      res = jnp.sum((d_lhs - rhs)**2)
      return res

  def make_step_scan(carry, points, optim, model):
      model_params, opt_state, grid, weights, wieghts_mc = carry
      l, grads = value_and_grad(lambda a, b, c: compute_time(a, b, c, model))(model_params, points, wieghts_mc)
      travel_time = compute_time(model_params, grid, weights, model)
      updates, opt_state = optim.update(grads, opt_state)
      model_params = optax.apply_updates(model_params, updates)
      return [model_params, opt_state, grid, weights, wieghts_mc], (l, travel_time)

  N = 512
  x0, x1 = 0, jnp.pi
  y0, y1 = 0, 2
  fine_weights, fine_grid = get_weights_and_grid(N, x0, x1, y0, y1)

  N = 128
  N_batch = 128
  N_updates = 30000
  x0, x1 = 0, jnp.pi
  y0, y1 = 0, 2
  weights, grid = get_weights_and_grid(N, x0, x1, y0, y1)

  model_ = lambda model_params, x: NN(model_params, x, x0, x1, y0, y1)
  key = random.PRNGKey(33)
  keys = random.split(key)
  N_features = [1, 100, 1]
  N_layers = 3
  model_params_ = init_NN(keys[0], N_features, N_layers)

  learning_rate = 1e-3
  optim = optax.adam(learning_rate=learning_rate)
  opt_state = optim.init(model_params_)

  make_step_scan_ = lambda a, b: make_step_scan(a, b, optim, model_)

  points = (x1 - x0)*random.uniform(keys[0], (N_updates, N_batch, 1)) + x0
  weights_mc = (x1 - x0)*jnp.ones((N_batch,)) / N_batch
  carry, (l, history) = scan(make_step_scan_, [model_params_, opt_state, grid, weights, weights_mc], points)
  model_params_ = carry[0]
  ```

  </div>
</details>

<img src="{{ page.asset_path }}learning_curve_mc.svg" width="450">

For Monte Carlo integration, the travel time remains stable. Why is it still smaller than the best possible travel time? The reason is that I used $128$ Legendre-Gauss points to approximate the travel time. If one increases the number of points, the travel time approaches the best one from below.

## Final thoughts and practical problems

Have you heard of the [paperclip maximizer](https://nickbostrom.com/ethics/ai)? Or perhaps [George Dantzig's](George Dantzig) computations for an [optimal diet](https://www.argmin.net/p/dietary-shapes)? Even better is [this fun story from Richard Feynman](https://youtu.be/ipRvjS7q1DI?si=414ktLilqmKP7WUh&t=924), where he explains the failing heuristics produced by [Douglas Lenat's](https://en.wikipedia.org/wiki/Douglas_Lenat) [Eurisko](https://en.wikipedia.org/wiki/Eurisko) program. All these results are examples of *specification gaming* (see this [dedicated list](https://docs.google.com/spreadsheets/d/e/2PACX-1vRPiprOaC3HsCf5Tuum8bRfzYUiKLRqJmbOoC-32JorNdfyTiRRsR7Ea5eWtvsWzuxo8bjOxCG84dAg/pubhtml) for more).

Specification gaming occurs when an optimization process converges to a solution that perfectly satisfies the stated problem, but in a way the designer never intended. The Deep Brachistochrone is a perfect example of this behavior. We started with a continuous minimization problem. Because it was too hard to program directly, we created an approximation. However, our discrete formulation failed to be a completely faithful representation of the initial continuous problem—leaving loopholes that an optimization algorithm could exploit. This only became obvious when we applied a sufficiently strong model (a neural network) capable of finding and exploiting that misspecification.

I would argue that our core issue wasn't the approximation of the integral itself, but rather the discrepancy between two specific sets: the set of functions we can accurately integrate, and the set of functions a neural network (with a given initialization and SGD) can approximate. While both sets contain good approximations of the cycloid, the second set also contains solutions that our integration method handles poorly. Among those are "solutions" that yield impossibly fast travel times—at least, according to our faulty integration. Can we integrate neural networks better? [The answer may be negative](https://arxiv.org/abs/2505.17751), though I suspect we can still do well enough for most practically relevant cases. For the time being, my advice is simple: **use neural networks only when you have to!** Save them for genuinely high-dimensional problems, or, you know, when you need one more paper to get your PhD.

The Brachistochrone problem has been solved for centuries, making this mostly an academic exercise. But are there practical modern problems where integration leads to specification gaming? Absolutely. Almost all the cases I know of come from [variational methods in quantum mechanics](https://en.wikipedia.org/wiki/Variational_method_(quantum_mechanics)). Today, researchers parameterize wavefunctions using [neural networks](https://deepmind.google/blog/ferminet-quantum-physics-and-chemistry-from-first-principles/) or [tensor neural networks](https://arxiv.org/abs/2207.02754). In the latter case, non-stochastic integration can be used, carrying the exact same risks down the road. A similar situation may occur when neural networks are used to solve Kohn-Sham equations via direct minimization. Ultimately, if your approximation scheme involves a neural network, you must be incredibly careful with your integral estimation.

{% if page.references %}
<hr>
<h3>References</h3>
<ol style="line-height: 1.5; padding-left: 20px;">
  {% for ref in page.references %}
    <li id="{{ ref.key }}" style="margin-bottom: 8px;">

      {% if ref.year %}
        {{ ref.author }} ({{ ref.year }}).
      {% else %}
        {{ ref.author }}.
      {% endif %}

      "{{ ref.title }}"

      {% if ref.file %}
        [<a href="{{ page.asset_path }}{{ ref.file }}" target="_blank">File</a>]
      {% endif %}

      {% if ref.url %}
        [<a href="{{ ref.url }}" target="_blank">Link</a>]
      {% endif %}

      {% if ref.note %}
        <div style="font-size: 0.95em; color: #555; font-style: italic; margin-top: 4px; padding-left: 10px; border-left: 2px solid #eee;">
          {{ ref.note }}
        </div>
      {% endif %}
    </li>
  {% endfor %}
</ol>
{% endif %}
