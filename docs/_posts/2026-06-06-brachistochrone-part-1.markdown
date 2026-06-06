---
layout: main
title:  "Deep brachistochrone"
date:   2026-06-06
asset_path: "/assets/2026-06-06-brachistochrone-part-1/"
references:
  - key: "weisstein"
    author: "Weisstein, Eric W."
    title: "Brachistochrone Problem."
    url: "https://mathworld.wolfram.com/BrachistochroneProblem.html"

  - key: "shafer2007"
    author: Shafer, Douglas S."
    year: 2007
    title: "The Brachistochrone: Historical Gateway to the Calculus of Variations."
    note: "I used this article to look up how to solve the Euler-Lagrange equations without resorting to inverse trigonometric functions. Reportedly, it took Newton an entire night to solve the problem, so I am not ashamed of this."

  - key: "sussmann1997"
    author: "Sussmann, Hector J., and Jan C. Willems."
    year: 1997
    title: "300 years of optimal control: from the brachystochrone to the maximum principle."
    url: "https://ieeexplore.ieee.org/document/588098/"

  - key: "kingma2014"
    author: "Kingma, Diederik P., and Jimmy Ba."
    year: 2014
    title: "Adam: A Method for Stochastic Optimization."
    url: "https://arxiv.org/abs/1412.6980"

  - key: "broer2014"
    author: "Broer, Henk W."
    year: 2014
    title: "Bernoulli's light ray solution of the brachistochrone problem through Hamilton's eyes."
    note: "I referenced this article in the section where the linear ansatz is integrated and it is claimed that this procedure can be continued to restore the entire Brachistochrone curve by introducing additional breaking points. The solution in the article is not exactly along these lines, but the first part (see Figure 3) is very close to this approach."
---

TL;DR: I define the famous Brachistochrone problem and solve it using modern Deep Learning methods. What could go wrong, right?

# Brachistochrone problem

Suppose we have a bead sliding along a wire that connects two points: $(0, 0)$ and $(x_1, y_1)$. We assume that there is no friction, the mass of the bead is $m$, and the gravitational force is $mg$. For convenience, we parametrize the shape of the wire as a function $f(x)$, where $x \in [0, x_1]$, and select coordinates as shown in the picture below.

<img src="{{ page.asset_path }}geometry.png" width="300">

The Brachistochrone problem asks about the *shape of the wire*: how do we select $f(x)$ such that the bead, starting from $(0, 0)$ with zero velocity, reaches $(x_1, f(x_1))$ as fast as possible? In other words, we want to minimize the travel time of the bead.

As demonstrated below, the travel time can be expressed as an integral:

$$
\int_{0}^{x_1}\frac{\sqrt{1 + \left(f^{'}(u)\right)^2}du}{\sqrt{f(u)}} = \sqrt{2g} t.
$$

<details>
  <summary><b>Deriving the travel time of the bead using Lagrangian Mechanics</b></summary>

  <div style="margin-top: 10px; padding-left: 15px; border-left: 2px solid #ccc;">
    <p>
    The standard way to find the equations of motion is to write down the Lagrangian function. The position of the bead on the wire is:
    $$
    \vec{r}(t) = x(t)\vec{e}_x + f(x(t)) \vec{e}_{y},
    $$
    so we can write the Lagrangian:
    $$
    L(x, \dot{x}) = \frac{m\left(1 + \left(f^{'}(x)\right)^2\right)\left(\dot{x}\right)^2}{2} - mgf(x) \equiv \text{kinetic energy} - \text{potential energy}.
    $$
    The equations of motion are:
    $$
    m\left(1 + \left(f^{'}(x)\right)^2\right)\ddot{x}(t) = mf^{''}(x)f^{'}(x) \left(\dot{x}(t)\right)^2 + mgf^{'}(x).
    $$
    Since the system's energy is conserved ($L$ does not explicitly depend on time), we can multiply by $\dot{x}$ and observe that the equations of motion reduce to:
    $$
    \frac{d}{dt} \left(\left(1 + \left(f^{'}(x)\right)^2\right)\frac{\left(\dot{x}(t)\right)^2}{2} - gf(x)\right) = 0.
    $$
    We are interested in the scenario where the bead is initially at rest ($\dot{x}(0) = 0$), and since $f(0) = 0$, we have:
    $$
    \left(1 + \left(f^{'}(x)\right)^2\right)\frac{\left(\dot{x}(t)\right)^2}{2} = gf(x).
    $$
    The trajectory $x(t)$ can then be given implicitly in the form of an integral:
    $$
    \int_{0}^{x}\frac{\sqrt{1 + \left(f^{'}(u)\right)^2}du}{\sqrt{f(u)}} = \sqrt{2g} t.
    $$
    </p>
  </div>
</details>

# Solution with the calculus of variations

Formally, we are interested in finding a smooth function that solves the optimization problem:

$$
\inf_{f(x)}\int_{0}^{x_1}\frac{\sqrt{1 + \left(f^{'}(u)\right)^2}du}{\sqrt{f(u)}} \text{ s.t. }  f \in C^{(1)}([0, x_1]).
$$

When the problem was initially posed by [Johann Bernoulli in 1696](https://en.wikipedia.org/wiki/Brachistochrone_curve#Introduction_of_the_problem), it was on the cutting edge of mathematics and later turned out to be important to many branches of the natural sciences ([Sussmann, Willems, 1997](#sussmann1997)). Nowadays, the standard solution technique, the [Euler-Lagrange equations](https://en.wikipedia.org/wiki/Euler%E2%80%93Lagrange_equation), can be found everywhere. Following ([Shafer, 2007](#shafer2007)), I derive the general form of the Brachistochrone curve below.

<details>
  <summary><b>Solution to the Brachistochrone problem by the Calculus of Variations</b></summary>
  <div style="margin-top: 10px; padding-left: 15px; border-left: 2px solid #ccc;">
    <p>
    Recall that for a general functional that depends on a function $f(u)$ and its derivative $L\left(f, f'\right)$, the necessary condition for optimality is:
    $$
    \frac{d}{du} \left(\frac{\partial L\left(f, f^{'}\right)}{\partial f^{'}}\right) - \frac{\partial L\left(f, f^{'}\right)}{\partial f} = 0.
    $$

    In our case, this condition reads
    $$
    \frac{d}{du} \frac{f^{'}(u)}{\sqrt{f(u)(1 + \left(f^{'}(u)\right)^2)}} = -\frac{1}{2}\frac{\sqrt{(1 + \left(f^{'}(u)\right)^2)}}{\left(f(u)\right)^{3/2}}.
    $$
    It is more convenient to use energy conservation. Recall that if the Lagrangian function does not depend on $u$ explicitly, the energy remains constant along the trajectories:
    $$
    \begin{split}
    &\frac{d}{du} \underbrace{\left(f^{'}(u) \frac{\partial L\left(f, f^{'}\right)}{\partial f^{'}} - L\left(f, f^{'}\right)\right)}_{\text{energy}} = f^{''}(u)\frac{\partial L\left(f, f^{'}\right)}{\partial f^{'}} + f^{'}(u)\frac{d}{du} \left(\frac{\partial L\left(f, f^{'}\right)}{\partial f^{'}}\right)\\ &- f^{'}(u)\frac{\partial L\left(f, f^{'}\right)}{\partial f} - f^{''}(u)\frac{\partial L\left(f, f^{'}\right)}{\partial f^{'}} = f^{'}(u)\underbrace{\left(\frac{d}{du} \left(\frac{\partial L\left(f, f^{'}\right)}{\partial f^{'}}\right) - \frac{\partial L\left(f, f^{'}\right)}{\partial f}\right)}_{\text{Euler-Lagrange equation}} = 0.
    \end{split}
    $$
    For the Brachistochrone problem, energy conservation reduces to a first-order ordinary differential equation:
    $$
    f(u) \left(1 + \left(\frac{d }{d u} f(u)\right)^2\right) = C.
    $$
    This equation can be solved explicitly by separation of variables, but we will cheat a little bit and define a new variable, $\frac{d f}{ du} = -\tan \theta$. Using this variable, we get:
    $$
    f(\theta) = C \left(\cos\theta\right)^2 = \frac{1}{2}C(1 + \cos \left(2\theta)\right),
    $$
    $$
    \frac{d u}{d \theta} = \frac{d u }{d f} \frac{df}{d\theta} = C\frac{\cos\theta}{\sin\theta} \sin(2\theta) = 2 C\left(\cos\theta\right)^2 = C(1 + \cos\left(2\theta\right)).
    $$
    Integrating the second equation gives:
    $$
    u(\theta) = C \theta  \frac{C}{2}\sin\left(2\theta\right) + C_{0}.
    $$
    Equivalently, we can write:
    $$
    \begin{split}
    &f(v) = C\left(1 + \cos(v)\right),\\
    &u(v) = Cv + C \sin(v) + C_0.
    \end{split}
    $$
    We select $v=-\pi$ as the starting point. To ensure the curve starts from the origin $(0, 0)$, we fix $C_0 = \pi C$.

    The second condition is that the curve must pass through the target point $(x_1, y_1)$:
    $$
    \begin{split}
    &f(v_1) = y_1 = C\left(1 + \cos(v_1)\right),\\
    &u(v_1) = x_1 = C\left(v_1 + \sin(v_1) + \pi\right).
    \end{split}
    $$
    To find $v_1$ we solve
    $$
    g(v_1) = \frac{v_1 + \sin(v_1) + \pi}{1 + \cos\left(v_1\right)} = \frac{x_1}{y_1} > 0.
    $$
    A unique solution always exists because the expression on the left, viewed as a function of $v_1$, has the following properties: (i) $g(-\pi) = 0$; (ii) $\lim_{x\rightarrow\pi}g(x) = +\infty$ (from the left); (iii) $g'(x) > 0$ for $x \in (-\pi, \pi)$.

    To prove the last property, we take the derivative:
    $$
    g^{'}(x) = \frac{\left(1 + \cos(x)\right)^2 + (x + \sin(x) + \pi)\sin(x)}{\left(1 + \cos(x)\right)^2} = \frac{2(1 + \cos(x)) + \pi(\frac{x}{\pi}\sin(x) + 1)}{\left(1 + \cos(x)\right)^2} > 0.
    $$
    Once $v_1$ is found as the solution to the equation, we determine $C$ using:
    $$
    C = \frac{y_1}{1 + \cos(v_1)}.
    $$
    </p>
  </div>
</details>

For convenience, we will use a particular solution that passes through the points $(0, 0)$ and $(\pi, 2)$. We introduce an additional variable $v \in [-\pi, 0]$ and write the solution in the parametric form:

$$
  \begin{split}
    &f(v) = 1 + \cos(v),\\
    &x(v) = v + \sin(v) + \pi.
  \end{split}
$$

This curve is known as a [cycloid](https://en.wikipedia.org/wiki/Cycloid), and the travel time of the bead is given by:

$$
\int_{0}^{\pi}\frac{\sqrt{1 + \left(f^{'}(u)\right)^2}du}{\sqrt{f(u)}} = \int_{0}^{x}\frac{1}{f(u)}du = \int_{-\pi}^{0}\frac{1 + \cos(v)}{1 + \cos(v)}dv = \pi = \sqrt{2g} t.
$$

Interestingly, a bead attached to a cycloid can actually move upward during part of its trajectory, yet the travel time remains the absolute shortest among any possible curve ([Weisstein](#weisstein)). We will see about that!

# Solution using parametric curves and optimization

We should not expect to get lucky all the time, as closed-form solutions are rarely available. Instead, a more pragmatic approach is to resort to numerical optimization:
1. We select a parametric model $f(x;\theta)$ such that $f(0;\theta) = 0$ and $f(\pi;\theta) = 2$ for all $\theta$.
2. We approximate the integral using a numerical quadrature rule: $\int_{0}^{x_1} \psi(x)dx \simeq \sum_{j=1}^{M} \psi(x_j) w_j$. I will use [Gauss-Legendre quadrature](https://en.wikipedia.org/wiki/Gauss%E2%80%93Legendre_quadrature).
3. We approximate the derivative using automatic differentiation (AD). We resort to [JAX](https://docs.jax.dev/en/latest/index.html) for this.
4. To solve the resulting optimization problem $\min_{\theta}F(\theta)$, we calculate gradients via AD and apply a first-order optimization method. In our case, this will be Adam ([Kingma, Ba, 2014](#kingma2014)).

This scheme can be implemented with just a few lines of code:

```python
import jax.numpy as jnp
import optax

from scipy.special import roots_legendre
from jax.lax import scan
from jax.nn import gelu
from jax import grad, value_and_grad, random, vmap

def get_weights_and_grid(N, x0, x1, y0, y1):
    grid, weights = roots_legendre(N_x)
    grid = jnp.array((x1 - x0) * (grid + 1) / 2 + x0).reshape(-1, 1)
    weights = (x1 - x0) * jnp.array(weights) / 2
    return weights, grid

def compute_time(model_params, grid, weights, model):
    dy = vmap(grad(lambda x: model(model_params, x)[0]))(grid)[:, 0]
    y = vmap(model, in_axes=(None, 0))(model_params, grid)[:, 0]
    S = jnp.sqrt(1 + dy**2) / jnp.sqrt(jnp.abs(y))
    res = jnp.sum(weights * S)
    return res

def make_step_scan(carry, ind, optim, model):
    model_params, opt_state, grid, weights = carry
    l, grads = value_and_grad(lambda a, b, c: compute_time(a, b, c, model))(model_params, grid, weights)
    updates, opt_state = optim.update(grads, opt_state)
    model_params = optax.apply_updates(model_params, updates)
    return [model_params, opt_state, grid, weights], l
```

To use this framework, one needs to implement a function that evaluates the parametric model $f(x;\theta)$. How to do this is illustrated below.

## The straight line

A straight line, $f(x) = \frac{2}{\pi}x$, has no free parameters, so there is nothing to optimize. Its travel time can also be explicitly computed:

$$
\int_{0}^{\pi}\frac{\sqrt{1 + \left(f^{'}(u)\right)^2}du}{\sqrt{f(u)}} = \sqrt{\frac{2}{\pi} + \frac{\pi}{2}}\int_{0}^{\pi} \frac{du}{\sqrt{u}} =  2\sqrt{\pi}\sqrt{\frac{2}{\pi} + \frac{\pi}{2}} = \sqrt{2gt}.
$$

By introducing extra breaking points and evaluating the integral on each interval, one can reconstruct the cycloid solution ([Broer, 2014](#broer2014)).

## The quadratic curve

This is our first non-trivial example. The curve has one free parameter, $\kappa$: $f(x; \kappa) = \frac{2}{\pi}x + \kappa x(x - \pi)$. In code, this parametric curve reads:

```python
def quadratic_model(model_params, x, x0, x1, y0, y1):
    y = y0 * (x - x1) / (x0 - x1) + y1 * (x - x0) / (x1 - x0) + model_params[0] * (x - x0) * (x - x1)
    return y
```

Below is the implementation of the optimization process:
```python
N = 128
x0, x1 = 0, jnp.pi
y0, y1 = 0, 2
weights, grid = get_weights_and_grid(N, x0, x1, y0, y1)

model_ = lambda model_params, x: quadratic_model(model_params, x, x0, x1, y0, y1)
model_params_ = jnp.array([0.0, ])

learning_rate = 1e-3
optim = optax.adam(learning_rate=learning_rate)
opt_state = optim.init(model_params_)

make_step_scan_ = lambda a, b: make_step_scan(a, b, optim, model_)

N_updates = 5000
ind = jnp.arange(N_updates)
carry, history = scan(make_step_scan_, [model_params_, opt_state, grid, weights], ind)
model_params_ = carry[0]

model_time = compute_time(model_params_, grid, weights, model_)
print("time", model_time)
```

This code outputs: `time 4.5505657`.

## The cubic curve

The cubic curve has two parameters, $\alpha$ and $\beta$: $f(x; \alpha, \beta) = \frac{2}{\pi}x + x(x - \pi) (\alpha x + \beta)$.

```python
def cubic_model(model_params, x, x0, x1, y0, y1):
    y = y0 * (x - x1) / (x0 - x1) + y1 * (x - x0) / (x1 - x0) + (x - x0) * (x - x1) * (model_params[0] * x + model_params[1])
    return y
```

The code for optimization requires only a minor adjustment:
```python
...

model_ = lambda model_params, x: cubic_model(model_params, x, x0, x1, y0, y1)
model_params_ = jnp.array([0.0, 0.0])

...
```
The travel time we found for the cubic curve is: `time 4.4826336`. The result is slightly better than the quadratic curve! Maybe we can do even better with more parameters?

## The Neural Network

Our ultimate parametric model—a universal approximator *(wink)*—is a feedforward neural network. The unconstrained parametric expression is:

$$
\begin{split}
v_0 &= x;\\
v_{i+1} &= \sigma\left(W_{i}v_{i} + b_{i}\right), i = 1,\dots, N-2;\\
v_{N} &= W_{N-1}v_{N-1} + b_{N-1} =: \phi(x),\\
\end{split}
$$

where $\sigma$ is a non-linear, smooth activation function, and $W_{i}$ and $b_{i}$ are matrices and vectors of conformable sizes. The constrained version is $f(x; \theta) = \frac{2}{\pi} x + x(x - \pi)\phi(x).$ In code, we implement two functions: one for initializing all the matrices and vectors, and one for evaluating the model.

```python
def init_NN(key, N_features, N_layers):
    N_in, N_pr, N_out = N_features
    N_features_ = [N_in,] + [N_pr,]*N_layers + [N_out,]
    model = dict()
    model['b'] = [jnp.zeros((n_out,)) + 0.1 for n_in, n_out in zip(N_features_[:-1], N_features_[1:])]
    model['W'] = [2 * random.normal(key, (n_out, n_in)) / jnp.sqrt(n_out + n_in) for n_in, n_out, key in zip(N_features_[:-1], N_features_[1:], random.split(key, N_layers+1))]
    return model

def NN(model_params, x, x0, x1, y0, y1):
    y = model_params['W'][0] @ x + model_params['b'][0]
    for w, b in zip(model_params['W'][1:], model_params['b'][1:]):
        y = gelu(y)
        y = w @ y + b
    y_ = y0 * (x - x1) / (x0 - x1) + y1 * (x - x0) / (x1 - x0) + y * (x - x0) * (x - x1)
    return y_
```

The adjustments for the optimization code are:
```python
...

model_ = lambda model_params, x: NN(model_params, x, x0, x1, y0, y1)
key = random.PRNGKey(33)
N_features = [1, 100, 1]
N_layers = 3
model_params_ = init_NN(key, N_features, N_layers)

...

N_updates = 10000

...
```

With that many iterations, we got `time 2.2218027`. Wow, that is great! Surely, we can drop it even more with more parameters and iterations.

# Leaderboard

All the travel times we found for the Brachistochrone problem are summarized in the table below:

|curve| $\sqrt{2g}t$|
|:---:|:---:|
|straight line|$5.266$|
|quadratic|$4.551$|
|cubic|$4.483$|
|cycloid|$3.142$|
|neural network (ours)|${\bf 2.221}$|
{: style="--table-padding: 40px;"}

Now our table is of production quality. You could take it and submit it to NeurIPS right away! In the exact same way that Deep Learning has "revolutionized the solving of partial differential equations" ([as they write in Nature](https://www.nature.com/articles/s42256-026-01233-9)), it has now made the fastest descent just a little bit faster.

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
