# Theory: dense predictions evaluated through line integrals

Appendix-ready statements and proofs. Every claim is checked numerically in `notebooks/s2_synthetic_benchmark.ipynb`.

## 0. Setting

A volume of voxels $i$ carries a reference value $y_i$ (HU), a prediction $\hat y_i$, and a source image $x$.
Stopping power is $g(y)$, with $g:\mathbb R\to\mathbb R$ continuous, non-decreasing and piecewise linear
(pyRadPlan's default table). The SPR error is $\varepsilon_i = g(\hat y_i) - g(y_i)$.

A **segment** $r$ is an ordered set of $n_r$ voxels along a beam direction $n$, sampled with step $\Delta s$,
from beam entry to the body exit (*chord*) or to the distal edge of the target. Its length is $L_r = n_r\Delta s$ and
the water-equivalent range error is

$$\Delta\mathrm{WET}_r \;=\; \Delta s\sum_{i\in r}\varepsilon_i .$$

This identity is exact; no linearisation of $g$ is involved.

## 1. Voxel-separable metrics cannot control range

**Proposition 1 (permutation blindness).** Let $M$ be any metric that depends on the errors only through their
empirical distribution $\{\varepsilon_i\}$ (MAE, MSE, PSNR, error histograms). Fix the multiset of errors and a
segment $r$. Over all permutations $\pi$ of voxel positions, $M(\varepsilon\circ\pi)$ is constant while
$\Delta\mathrm{WET}_r(\varepsilon\circ\pi)$ takes every value $\Delta s\sum_{i\in S}\varepsilon_i$ with $|S|=n_r$,
from $\Delta s\,S_{\min}$ (sum of the $n_r$ smallest errors) to $\Delta s\,S_{\max}$ (sum of the $n_r$ largest).

*Proof.* $M$ is invariant by assumption. A permutation can place any $n_r$ of the errors on the voxels of $r$. $\square$

The metric therefore cannot distinguish range errors anywhere in $[\Delta s\,S_{\min},\Delta s\,S_{\max}]$. This interval
contains values of both signs if and only if $S_{\min}<0<S_{\max}$; errors of both signs are necessary but not
sufficient (errors $\{-1,5,5,5\}$ with $n_r=2$ give $S_{\min}=4>0$). When the condition holds, the metric is
uninformative about the sign; in all cases it is uninformative about the magnitude within that interval.

**Proposition 1′ (weighted rays, sharp bounds).** Let $w_i\ge0$ be any ray weighting over the $N$ body voxels
($w_i=\Delta s\,\mathbf 1[i\in r]$ for a segment, or a depth-dependent weighting, or a sum over overlapping rays), and
write $\varepsilon_{(1)}\le\dots\le\varepsilon_{(N)}$, $w_{(1)}\le\dots\le w_{(N)}$. Over all rearrangements $\pi$,

$$\sum_k \varepsilon_{(k)}w_{(N+1-k)} \;\le\; \sum_i w_i\,\varepsilon_{\pi(i)} \;\le\; \sum_k \varepsilon_{(k)}w_{(k)} .$$

The upper bound is attained exactly by arrangements that order errors like weights (comonotone), the lower one by
opposite orderings (antimonotone), and when neither sequence is constant no other arrangement attains them.

*Proof.* The inequality is the Hardy–Littlewood rearrangement inequality. Attainment and its exactness are the discrete
case of the sharp Fréchet–Hoeffding bound (Bellini, Koch-Medina, Munari and Svindland, 2022, Lemma A.2): for
non-constant $X,Y$, $\max_{X'\sim X}\mathbb E[X'Y]$ is reached exactly under comonotone coupling, which is what an
arrangement of $\varepsilon$ against $w$ is when both are read as random variables on the uniform law over voxels. $\square$

The worst-case range error compatible with a given error histogram is therefore the one that stacks the largest
errors on the most heavily weighted voxels: errors that are *spatially coherent along the beam*.
`sctrange.metrics.rearrangement_bounds` computes both bounds.

**Proposition 2 (autocorrelation decomposition).** Let $(\varepsilon_i)_{i\in r}$ be weakly stationary along the segment with
mean $\mu$, variance $\sigma^2$ and correlation $\rho(k)$. Then

$$\mathbb E\big[\Delta\mathrm{WET}_r^2\big] = L_r^2\mu^2 + L_r\sigma^2\ell_r,\qquad
\ell_r = \Delta s\Big[1+2\sum_{k=1}^{n_r-1}\Big(1-\frac{k}{n_r}\Big)\rho(k)\Big].$$

*Proof.* $\mathbb E[\Delta\mathrm{WET}_r]=\Delta s\,n_r\mu=L_r\mu$ and
$\mathrm{Var}(\Delta s\sum_i\varepsilon_i)=\Delta s^2\sigma^2\sum_{i,j}\rho(|i-j|)=\Delta s^2\sigma^2\big[n_r+2\sum_{k\ge1}(n_r-k)\rho(k)\big]=L_r\sigma^2\ell_r$. $\square$

**Corollary 2.1 ($\sqrt n$ gap at fixed MAE).** For centred Gaussian errors, $\mathrm{MAE}=\sigma\sqrt{2/\pi}$ whatever $\rho$.
The RMS range error ranges from $\sigma\sqrt{L_r\Delta s}$ (white errors, $\ell_r=\Delta s$) to $\sigma L_r$ ($\rho\equiv1$):
a factor $\sqrt{n_r}$, i.e. 10 for a 10 cm path at 1 mm. Neither end is a hard bound: negatively correlated errors
(e.g. alternating over- and under-estimation across an edge) give $\ell_r<\Delta s$ and an RMS below the white level, down
to zero for exactly cancelling errors. Range error is controlled by $\ell_r$, which MAE does not constrain in either direction.

## 2. Spectral weighting (Result A)

**Proposition 3 (exact discrete identity).** On a grid that is periodic along the beam axis with $N$ samples, let
$W_p=\Delta s\sum_{j=0}^{L-1}\varepsilon_{p+j}$ (indices mod $N$, lateral position fixed). With $\hat\varepsilon$ the DFT over all axes
and $N_{\mathrm{tot}}$ the number of voxels,

$$\sum_p W_p^2 = \frac{\Delta s^2}{N_{\mathrm{tot}}}\sum_k |\hat\varepsilon_k|^2\, D_L(f_k)^2,\qquad
D_L(f)=\frac{\sin(\pi f L)}{\sin(\pi f)},$$

where $f_k$ is the frequency **along the beam** (cycles per sample). The weight does not depend on lateral frequency.

*Proof.* $W$ is the circular cross-correlation of $\varepsilon$ with a box of $L$ ones along the beam axis; its DFT is
$\hat\varepsilon_k\overline{B(f_k)}$ with $|B(f)|=|D_L(f)|$. Parseval's identity for the unnormalised DFT gives the result. $\square$

The normalised weight $D_L^2/L^2$ equals 1 at $f=0$, vanishes at $f=m/L$, and is bounded by $1/(L\sin\pi f)^2\approx 1/(\pi f L)^2$:
**range error is low-pass along the beam with cut-off $\sim 1/L$, and all-pass across it.** By comparison, voxel-wise
squared error weights all frequencies equally and a forward-difference gradient loss weights them by
$4\sin^2(\pi f)$, increasing towards Nyquist, so a gradient term moves the training objective away from the band that
range integrates. For SSIM, perceptual and adversarial losses no transfer function of this kind exists (they are
non-quadratic, local or learned), and we make no spectral claim about them: whether they shift error energy towards or
away from low beam-axis frequencies is an empirical hypothesis, tested by the beam-aligned spectra of S3.

*Scope.* The identity is exact for circular segments on a periodic grid. Segments clipped to a finite body (entry to exit,
or entry to distal target) are windowed, which convolves the spectrum with the window's transform and couples
frequencies near $f=0$. The low-pass character survives (the window only adds leakage of width $\sim 1/L$), but the
formula is not an identity there; the S2 checks use periodic segments, and S3 measures spectra on Hann-tapered windows
inside the body rather than relying on the formula.

**Proposition 3.1 (full traversal, all angles).** For a field $e$ on $\mathbb R^2$ and parallel projections
$P_\theta e(s)=\int e(s\,u_\theta + t\,n_\theta)\,dt$ over $\theta\in[0,\pi)$,

$$\int_0^\pi\!\!\int |P_\theta e(s)|^2\,ds\,d\theta \;\propto\; \int_{\mathbb R^2}\frac{|\hat e(k)|^2}{|k|}\,dk .$$

*Proof.* By the Fourier slice theorem the 1-D transform of $P_\theta e$ at $\omega$ is $\hat e(\omega u_\theta)$, the slice
**perpendicular** to the beam. Parseval on each projection and the polar change of variables $k=\omega u_\theta$,
$dk=|\omega|\,d\omega\,d\theta$, give the result (constants depend on the Fourier convention). $\square$

For a single plane wave of frequency $|k|$ in a window of size $R$, only directions within $\sim 1/(|k|R)$ of the
wavefronts accumulate, so the energy summed over angles decays as $|k|^{-1}$. With a finite set of beams, full-traversal
range error sees only the slices $\{k : k\cdot n_\theta=0\}$; segment integrals of length $L$ broaden each slice to a
band of width $\sim 1/L$.

## 3. Bayes acts under range losses

**Proposition 4 (squared range loss).** For any family of segments $\mathcal R$ and the risk
$\mathbb E\big[\sum_{r\in\mathcal R}\Delta\mathrm{WET}_r^2\mid x\big]$, the voxel-wise prediction

$$\hat y_i^\star = g^{-1}\big(\mathbb E[g(Y_i)\mid x]\big)$$

is a Bayes act, simultaneously for every $\mathcal R$ (in particular for every treatment plan). If $\mathcal R$ contains, for
every voxel, two segments differing only by that voxel (e.g. segments from entry to every depth), the Bayes act is unique
in SPR, $g(\hat y_i)=\mathbb E[g(Y_i)\mid x]$.

*Proof.* With $W_r(\hat y)=\Delta s\sum_{i\in r}g(\hat y_i)$, the risk of segment $r$ is
$(W_r(\hat y)-\mathbb E[W_r(Y)\mid x])^2+\mathrm{Var}(W_r(Y)\mid x)$; the variance does not depend on $\hat y$.
The bias vanishes for every $r$ when $g(\hat y_i)=\mathbb E[g(Y_i)\mid x]$ for all $i$, which is attainable because $g$ is
continuous and $\mathbb E[g(Y_i)\mid x]$ lies in its range. Under the stated richness of $\mathcal R$, differences of
segment sums isolate each voxel, so zero bias on all segments forces the voxel-wise identity. $\square$

The excess risk of any other prediction on segment $r$ is exactly $B_r^2$ with
$B_r=\Delta s\sum_{i\in r}\big(g(\hat y_i)-\mathbb E[g(Y_i)\mid x]\big)$. **A generative model cannot beat this regressor on
mean squared range error.** Regressing the mean in HU instead incurs the Jensen gap
$g(\mathbb E Y_i)-\mathbb E g(Y_i)$, which vanishes where $g$ is affine on the support of the posterior and not across a
change of slope (interfaces, partial volume, uncertain presence of air or bone).

**Proposition 5 (absolute loss creates range bias).** A voxel-wise absolute-error loss, in HU or in SPR, has Bayes act
$\mathrm{med}(Y_i\mid x)$ (medians commute with the monotone $g$). Its range bias on segment $r$ is

$$B_r^{\mathrm{L1}}=\Delta s\sum_{i\in r}\Big(\mathrm{med}\,g(Y_i)-\mathbb E\,g(Y_i)\Big).$$

*Remark (two-tissue posterior).* If voxel $i$ is an alternative tissue with probability $p_i<1/2$ and the two components
are well separated, the median lies in the base component, so
$b_i\approx p_i\big(g(\mu_{\mathrm{base}})-g(\mu_{\mathrm{alt}})\big)$. The sign is fixed by the tissue pair: SPR is
**over**-estimated next to uncertain gas and **under**-estimated next to uncertain bone. Because uncertainty is spatially
coherent (gas pockets, bone edges), the $b_i$ add up along beams and $B_r$ grows with $\sum_{i\in r}p_i$.

## 4. Which point functional is consistent under projection (Result B)

A separable score $S(\hat y,y)=\sum_i s(\hat y_i,y_i)$ with $s$ consistent for a law-determined functional $T$ yields
voxel-wise predictions $T(\mathcal L(Y_i\mid x))$. The downstream quantity is a linear combination $W=\sum_i a_iY_i$.
We call $T$ **projection-consistent** if $T(\mathcal L(W))=\sum_i a_iT(\mathcal L(Y_i))$ for every joint law and weights.

**Theorem 6.** Let $\mathcal M$ be the laws on $\mathbb R$ with finite support and rational probabilities, and
$T:\mathcal M\to\mathbb R$ with $T(\delta_c)=c$. If $T(\mathcal L(Y_1+\dots+Y_m))=\sum_kT(\mathcal L(Y_k))$ for every
random vector with marginals and sum in $\mathcal M$, then $T(F)=\int y\,dF$ for all $F\in\mathcal M$.
If moreover $T$ is continuous for the Wasserstein-1 distance, $T$ is the mean on all laws with finite first moment.
Conversely, the mean is projection-consistent.

*Proof.* Write $F$ as the uniform law on a multiset $\{v_0,\dots,v_{m-1}\}$ (common denominator $m$). Let $U$ be uniform on
$\{0,\dots,m-1\}$ and $Y_k=v_{(U+k)\bmod m}$, $k=0,\dots,m-1$. Each $Y_k$ has law $F$ and $\sum_kY_k=\sum_jv_j=m\,\bar v$
is constant. Hence $m\,\bar v=T(\delta_{m\bar v})=\sum_kT(F)=m\,T(F)$, so $T(F)=\bar v$. Laws with rational finite support
are $W_1$-dense in the laws with finite mean, which gives the extension. The converse is linearity of expectation. $\square$

The cyclic coupling is a joint mix in the sense of Wang and Wang (2016).

*Relation to the literature; no novelty is claimed for Theorem 6 as a statistical statement.*

- The closest result is the equivalence between **dependence neutrality and risk neutrality** of Wang and Wu (2020):
  a preference over sums of risky components that is indifferent to their dependence structure is represented by the
  expectation. Theorem 6 is its specialisation to point functionals, with an elementary proof on finitely supported
  laws that needs neither continuity nor monotonicity there.
- The "collapse to the mean" theorems of Bellini, Koch-Medina, Munari and Svindland (2021, 2022) reach the same
  conclusion from **different, not weaker,** hypotheses. They work on an atomless probability space with law-invariant
  functionals on random variables, and ask for affinity or translation invariance along a **single** non-constant
  direction $Z$, $\varphi(X+tZ)=\varphi(X)+at$, together with **convexity or quasiconvexity and lower semicontinuity**
  (their Theorems 5.1–5.5). Their counterexamples show that the regularity cannot be dropped: a law-invariant capacity can
  be linear along one direction without collapsing (their Example F.2), and without quasiconvexity translation
  invariance along one direction does not suffice (Example D.2). Theorem 6 trades their regularity for additivity under
  **all** couplings, which is exactly what dense prediction delivers, since the joint law of the voxels is unknown and
  the downstream sum must be right for every one of them.
- In dense prediction the point functional is applied voxel by voxel to $\mathcal L(Y_i\mid x)$; law invariance is
  therefore automatic, and quantiles and expectiles, which are neither linear along any direction nor dependence
  neutral, fail.

**Corollary 6.1.** Quantiles (including the median) and expectiles with $\tau\ne1/2$ are not projection-consistent.
With the non-linear $g$, the downstream quantity is $\sum_ia_ig(Y_i)$; applying Theorem 6 to $g(Y_i)$, **the separable
target must be $\mathbb E[g(Y_i)\mid x]$**, which any Bregman score in SPR space elicits (squared error being the canonical
choice), and no score in HU space does unless $g$ is affine on the posterior support.

## 5. Tails of range error need the joint law

**Proposition 7.** For $n\ge2$ voxels with non-degenerate marginals, there exist joint laws with identical marginals
whose segment sums have different quantiles and expected shortfalls. Consequently no function of the voxel-wise
marginals, and in particular no separable point prediction, determines $\mathrm{VaR}_q$ or $\mathrm{ES}_q$ of
$\Delta\mathrm{WET}_r$.

*Proof.* The comonotonic coupling gives $\mathrm{VaR}_q(\sum_iY_i)=\sum_i\mathrm{VaR}_q(Y_i)$, while for marginals as in
Theorem 6 the cyclic coupling makes the sum constant; the two quantiles differ unless the marginals are degenerate.
A smooth family interpolating between independence and strong dependence (the Gaussian copula of §6) gives the same
conclusion with fixed, continuous marginals. $\square$

*Remark (tail objectives are genuinely non-linear).* Let $\rho$ be a consistent risk measure applied to the law of the
range error, that is cash-additive, monotone for second-order stochastic dominance, normalised and lower semicontinuous
(Expected Shortfall is one). By Bellini et al. (2022, Theorem 5.7), if $\rho(tZ)=t\rho(Z)$ for all real $t$ and a single
non-constant $Z$, then $\rho$ is the expectation. A risk-averse tail objective on range error is therefore never linear
along any non-constant direction: it cannot be reduced to a sum of voxel-wise terms, which is the functional counterpart
of Proposition 7.

Two remedies follow. A generative model of the joint posterior provides the law of $\Delta\mathrm{WET}_r$ by sampling.
Alternatively, $(\mathrm{VaR}_q,\mathrm{ES}_q)$ of the range error is jointly elicitable (Fissler and Ziegel, 2016) and can be
learnt directly with a strictly consistent score; `sctrange.losses.fz_upper_tail_loss` implements the FZ0 member
(Patton, Ziegel and Chen, 2019) for the upper tail of $|\Delta\mathrm{WET}|$.

## 6. Proper evaluation of generative predictions

**Proposition 8 (projected CRPS).** For a predictive law $F$ of the field and an observation $y$, let
$\mathrm{PCRPS}(F,y)=\sum_{r\in\mathcal R}\mathrm{CRPS}\big(F\circ W_r^{-1},W_r(y)\big)$. PCRPS is proper. Two predictive
laws with identical voxel marginals receive identical expected voxel-wise CRPS, whereas their PCRPS differ whenever
their segment-sum laws differ. Over all linear functionals PCRPS is strictly proper on laws with finite first moment
(Cramér-Wold); over a finite family of segments it is proper but not strictly proper.

*Proof.* Each term is the CRPS of a one-dimensional law, which is strictly proper on laws with finite mean, so the sum is
proper. Expected voxel-wise CRPS depends on $F$ only through its marginals. If all one-dimensional projections of two
laws agree, the laws agree (Cramér-Wold), which yields strict propriety over all linear functionals. $\square$

## 7. Numerical verification map

| Claim | Notebook section | Check |
|---|---|---|
| Propositions 1, 1′ | 1 | exhaustive search over rearrangements equals the sorted bounds, attained only at sorted orders |
| Proposition 2, Corollary 2.1 | 1 | observed vs predicted RMS at fixed MAE, ratio white/correlated |
| Proposition 3 | 2 | Parseval identity to machine precision; measured vs theoretical transfer |
| Proposition 3.1 | 2 | slope of angle-summed energy vs frequency close to −1 |
| Propositions 4, 5 | 3 | Monte Carlo risk = analytic bias² + variance; estimator ranking; bias signs for gas and bone |
| Theorem 6, Corollary 6.1 | 4 | cyclic coupling: mean additive, median and expectiles not |
| Proposition 7 | 5 | identical marginals and Bayes act, different VaR/ES of range error |
| FZ score | 5 | grid minimiser at empirical VaR/ES |
| Proposition 8 | 6 | voxel CRPS cannot separate dependence; projected CRPS can |

## References

- Bellini F., Koch-Medina P., Munari C., Svindland G. Law-invariant functionals that collapse to the mean.
  *Insurance: Mathematics and Economics* 98 (2021) 83–91.
- Bellini F., Koch-Medina P., Munari C., Svindland G. Law-invariant functionals that collapse to the mean: beyond
  convexity. *Mathematics and Financial Economics* 16 (2022). doi:10.1007/s11579-022-00313-9; arXiv:2106.01281.
- Wang R., Wu Q. Dependence and risk attitudes: an equivalence. SSRN working paper (2020). doi:10.2139/ssrn.3707709.
- Fissler T., Ziegel J. F. Higher order elicitability and Osband's principle. *Annals of Statistics* 44 (2016).
- Gneiting T. Making and evaluating point forecasts. *JASA* 106 (2011).
- Gneiting T., Raftery A. E. Strictly proper scoring rules, prediction, and estimation. *JASA* 102 (2007).
- Patton A. J., Ziegel J. F., Chen R. Dynamic semiparametric models for expected shortfall (and value-at-risk).
  *Journal of Econometrics* 211 (2019).
- Wang B., Wang R. Joint mixability. *Mathematics of Operations Research* 41 (2016).
