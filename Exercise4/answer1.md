## Exercise1

The objective function maximized in the M-step is:
$$
Q (\theta) = \sum_ {n, k} r _ {n k} \left[ \log \pi_ {k} - \frac {1}{2} \left(x _ {n} - \mu_ {k}\right) ^ {T} \Sigma_ {k} ^ {- 1} \left(x _ {n} - \mu_ {k}\right) - \frac {D}{2} \log (2 \pi) - \frac {1}{2} \log \det \Sigma_ {k} \right]
$$
The optimal solution is known when there are no constraints.
$$
\Sigma_ {k} = \frac {1}{N _ {k}} \sum_ {n} r _ {n k} \left(x _ {n} - \mu_ {k}\right) \left(x _ {n} - \mu_ {k}\right) ^ {T}, \quad N _ {k} = \sum_ {n} r _ {n k}
$$

### Case1

$$
\Sigma_ {k} = \operatorname {d i a g} \left(s _ {k 1}, \dots , s _ {k D}\right)
$$

That is, each cluster has its own diagonal matrix where$ s_k∈R^D$ is the variance vector of each dimension.

Simplify the relevant terms in Q
$$
\det \left(\Sigma_ {k}\right) = \prod_ {d = 1} ^ {D} s _ {k d}, \quad \left(x _ {n} - \mu_ {k}\right) ^ {T} \Sigma_ {k} ^ {- 1} \left(x _ {n} - \mu_ {k}\right) = \sum_ {d = 1} ^ {D} \frac {\left(x _ {n d} - \mu_ {k d}\right) ^ {2}}{s _ {k d}}
$$
So the terms in Q involving skd are:
$$
Q \supset \sum_ {n} r _ {n k} \left[ - \frac {1}{2} \sum_ {d} \frac {\left(x _ {n d} - \mu_ {k d}\right) ^ {2}}{s _ {k d}} - \frac {1}{2} \sum_ {d} \log s _ {k d} \right]
$$
Take the partial derivative with respect to skd and set to zero
$$
\frac {\partial Q}{\partial s _ {k d}} = \sum_ {n} r _ {n k} \left[ \frac {\left(x _ {n d} - \mu_ {k d}\right) ^ {2}}{2 s _ {k d} ^ {2}} - \frac {1}{2 s _ {k d}} \right] = 0
$$

$$
\sum_ {n} r _ {n k} \left(x _ {n d} - \mu_ {k d}\right) ^ {2} = s _ {k d} \sum_ {n} r _ {n k} = s _ {k d} \cdot N _ {k}
$$

update rule
$$
s _ {k d} = \frac {1}{N _ {k}} \sum_ {n} r _ {n k} \left(x _ {n d} - \mu_ {k d}\right) ^ {2}, \quad \Sigma_ {k} = \operatorname {d i a g} \left(s _ {k}\right)
$$

### Case2

All clusters share a single full covariance matrix Σ

Terms in Q involving Σ
$$
Q \supset \sum_{n,k} r_{n k} \left[ - \frac{1}{2} \left(x_{n} - \mu_{k}\right)^{T} \Sigma^{-1} \left(x_{n} - \mu_{k}\right) - \frac{1}{2} \log \det \Sigma \right]
$$
Take the gradient with respect to Σ
$$
\nabla_ {\Sigma} Q = \frac {1}{2} \sum_ {n, k} r _ {n k} \left[ \Sigma^ {- 1} \left(x _ {n} - \mu_ {k}\right) \left(x _ {n} - \mu_ {k}\right) ^ {T} \Sigma^ {- 1} - \Sigma^ {- 1} \right] = 0
$$
Multiply both sides on the left by Σand on the right by Σ
$$
\sum_ {n, k} r _ {n k} \left(x _ {n} - \mu_ {k}\right) \left(x _ {n} - \mu_ {k}\right) ^ {T} = \Sigma \cdot \sum_ {n, k} r _ {n k} = \Sigma \cdot N
$$

since

$$
\sum_{n,k} r_{nk}=\sum_{k} N_{k}=N.
$$

Update rule:
$$
\Sigma = \frac {1}{N} \sum_ {n, k} r _ {n k} \left(x _ {n} - \mu_ {k}\right) \left(x _ {n} - \mu_ {k}\right) ^ {T}
$$

### Case3

All clusters share a single scalar variance$ σ^2$, giving $Σk=σ^2I$

Simplify:
$$
\Sigma^ {- 1} = \frac {1}{\sigma^ {2}} I, \quad \det \left(\sigma^ {2} I\right) = \left(\sigma^ {2}\right) ^ {D}
$$
Substituting into Q
$$
Q \supset \sum_ {n, k} r _ {n k} \left[ - \frac {1}{2 \sigma^ {2}} \| x _ {n} - \mu_ {k} \| ^ {2} - \frac {D}{2} \log \sigma^ {2} \right]
$$
Take the derivative with respect to $σ^2$ and set to zero
$$
\frac {\partial Q}{\partial \sigma^ {2}} = \sum_ {n, k} r _ {n k} \left[ \frac {\| x _ {n} - \mu_ {k} \| ^ {2}}{2 \left(\sigma^ {2}\right) ^ {2}} - \frac {D}{2 \sigma^ {2}} \right] = 0
$$
Multiplying both sides by $2(σ^2)^2$
$$
\sum_ {n, k} r _ {n k} \| x _ {n} - \mu_ {k} \| ^ {2} = \sigma^ {2} \cdot D \cdot \sum_ {n, k} r _ {n k} = \sigma^ {2} \cdot D \cdot N
$$
update rule
$$
\sigma^ {2} = \frac {1}{N D} \sum_ {n, k} r _ {n k} \| x _ {n} - \mu_ {k} \| ^ {2}, \quad \Sigma_ {k} = \sigma^ {2} I
$$

### Effect on the Other M-step Updates

**The parameterization of $\Sigma$ does not affect the updates for $\pi_k$ or $\mu_k$.**

- **$\pi_k$:** Obtained by maximizing $\sum_{n,k} r_{nk} \log \pi_k$ subject to $\sum_k \pi_k = 1$ via a Lagrange multiplier. This term contains only $\log \pi_k$ — no $\Sigma_k$ appears — so the update $\pi_k = N_k / N$ is identical in all three cases.

- **$\mu_k$:** The gradient is $\nabla_{\mu_k} Q = -\Sigma_k^{-1} \sum_n r_{nk}(x_n - \mu_k) = 0$. As long as $\Sigma_k$ is invertible, $\Sigma_k^{-1}$ can be factored out, and the update reduces to:

$$\mu_k = \frac{1}{N_k}\sum_n r_{nk} x_n$$

regardless of the form of $\Sigma_k$. All three constraints preserve invertibility ($s_{kd} > 0$, $\Sigma \succ 0$, $\sigma^2 > 0$), so this holds in every case.

**In summary**, the three constraints only change how the covariance is updated in the M-step, leaving $\pi_k$ and $\mu_k$ unchanged.
