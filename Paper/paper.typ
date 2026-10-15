#import "@preview/ijimai:3.0.0": *

#show: ijimai.with(
  config: toml("paper.toml"),
  bibliography: "bibliography.yaml",
  read: path => read-raw(path),
)

// ============================================================
// Layout and notation helpers
// ============================================================

#let MTE = [MTE]
#let SMTE = [SMTE]
#let GMTE = [GMTE]
#let EBM = [EBM]
#let RuleFit = [RuleFit]
#let XGBoost = [XGBoost]
#let CatBoost = [CatBoost]

// Helper for figures/tables that should appear inline and fit in one column.
#let here-figure(body, caption) = figure(
  body,
  caption: caption,
  placement: none,
)

#let here-table(body, caption) = figure(
  body,
  caption: caption,
  placement: none,
)

// Helper for wide figures/tables that should span two columns and float.
#let wide-figure(body, caption) = figure(
  body,
  caption: caption,
  placement: top,
  scope: "parent",
)

#let wide-table(body, caption) = figure(
  body,
  caption: caption,
  placement: top,
  scope: "parent",
)

#let polylog = $op("polylog")$

#let compact-table(body) = text(size: 6.5pt)[#body]
#let xcompact-table(body) = text(size: 6.0pt)[#body]

#let paper-image(path, width: 100%) = image(path, width: width)

// Independent statement counters
#let assumption-counter = counter("assumption")
#let proposition-counter = counter("proposition")
#let theorem-counter = counter("theorem")
#let corollary-counter = counter("corollary")

#let assumption(title, body) = {
  assumption-counter.step()
  block[
    #set par(justify: true)
    #strong[Assumption #context assumption-counter.display("1")#if title != none [ (#title)].] #body
  ]
  v(10pt)
}

#let proposition(title, body) = {
  proposition-counter.step()
  block[
    #set par(justify: true)
    #strong[Proposition #context proposition-counter.display("1")#if title != none [ (#title)].] #body
  ]
  v(10pt)
}

#let theorem(title, body) = {
  theorem-counter.step()
  block[
    #set par(justify: true)
    #strong[Theorem #context theorem-counter.display("1")#if title != none [ (#title)].] #body
  ]
  v(10pt)
}

#let corollary(title, body) = {
  corollary-counter.step()
  block[
    #set par(justify: true)
    #strong[Corollary #context corollary-counter.display("1")#if title != none [ (#title)].] #body
  ]
  v(10pt)
}

// ============================================================
// Main manuscript
// ============================================================

= Introduction

Tabular prediction is central to high-stakes domains such as medicine, finance, engineering, and public policy. In these applications, predictive accuracy alone is often insufficient. Practitioners require transparency to understand feature interactions, inspect decision boundaries, and assess whether learned representations align with domain knowledge @rudin2019.

Gradient-boosted decision trees (GBDTs), such as #XGBoost, LightGBM, and #CatBoost, are standard tools for tabular prediction @chen2016xgboost @ke2017lightgbm @prokhorenkova2018catboost @grinsztajn2022why. However, their decision functions often consist of large ensembles of trees, making global interpretation difficult. Post-hoc explanation frameworks such as SHAP and LIME provide local attribution summaries @ribeiro2016 @lundberg2017shap, but they do not expose a compact global rule structure equivalent to the trained ensemble. Conversely, classical glassbox models—such as sparse linear regression, shallow decision trees @breiman1984, and rule lists @letham2015interpretable—offer explicit transparency but often lack the capacity to capture complex, high-order feature interactions.

Rule-based models combine transparency with the ability to model interactions via conjunctions. A rule of the form

$ "IF " x_1 > c_1 " AND " x_3 < c_3 " THEN predict " y $ <eq-rule-example>

Such a rule (@eq-rule-example) directly exposes its underlying logic and simplifies local auditing. However, learning optimal rules is traditionally a combinatorial, non-differentiable problem. #RuleFit @friedman2008rulefit bypasses this by extracting rules heuristically from decision trees before fitting a sparse linear model, which decouples rule generation from coefficient optimization. Neural rule models and differentiable decision trees replace hard thresholds with continuous relaxations @irsoy2012 @kontschieder2015, but their internal representations can become dense or computationally expensive to extract.

This work introduces _Margin-Template Rule Ensembles_ (#MTE), a differentiable rule architecture designed to preserve the semantic clarity of rule-based models while allowing joint end-to-end optimization of rules and weights via gradient descent. Each rule is formulated using soft, axis-aligned literals with learnable thresholds, directions, evidence signs, and sparsity masks. Literal evidence is transformed into a signed log-margin, and a rule activates when its margin vector aligns with a learned probabilistic template.

We analyze two template formulations: a Gaussian template (#GMTE) and a Student-$t$ template (#SMTE). The Student-$t$ variant provides heavy-tailed, robust matching in margin space and theoretically bounds the influence of outlier residuals.

Performing rule matching in signed log-margin space decouples evidence supporting and opposing each literal, transforming their relative magnitude into a log-ratio. Rather than evaluating a simple product or sum of soft comparisons, an #MTE rule acts as a structured template over an evidence vector. This formulation allows rules to encode bidirectional evidence, encourage structural sparsity via masks, quantify tolerance through learned scales, and improve robustness through heavy-tailed energy functions.

== Research Questions

This study investigates the following research questions:

1. Can differentiable margin-template rules accurately recover underlying feature dependencies and interaction structures within synthetic environments?
2. How does a differentiable rule ensemble perform compared to strong tabular baselines on benchmark datasets?
3. Does the Student-$t$ template yield a theoretically verifiable robustness advantage over Gaussian margin matching?
4. Can the optimized continuous model be extracted into discrete, human-readable rules incorporating thresholds, signs, masks, and head weights?
5. How do the individual architectural mechanisms—signed evidence, log-margins, mask penalties, learnable thresholds, adaptive scales, and template choice—influence performance and interpretability?

== Contributions

The primary contributions of this work are:

- We introduce #MTE, a differentiable rule-ensemble architecture based on signed log-margin literals and probabilistic margin templates.
- We formulate two variants: #GMTE, using Gaussian templates, and #SMTE, using robust Student-$t$ templates.
- We establish theoretical foundations for #MTE, including smoothness, Lipschitz continuity bounds, hard-rule limit convergence, universal approximation rates, Rademacher complexity, and finite-sample generalization bounds. We also formally prove the bounded-influence property of the Student-$t$ formulation.
- We evaluate #MTE on controlled synthetic tasks to measure feature and interaction recovery, and benchmark the models against #EBM, #XGBoost, #CatBoost, and #RuleFit across 28 tabular benchmarks consisting of 25 real-world datasets and 3 synthetic generators.
- We conduct an ablation study of 15 configurations across five synthetic datasets and five random seeds, totaling 375 model runs, to isolate the impact of signed evidence, log-margins, mask penalties, learnable thresholds, adaptive scales, and template formulations.

#MTE does not seek to universally outperform highly optimized gradient-boosted trees in raw predictive accuracy. Instead, it addresses a distinct design goal: achieving competitive performance through end-to-end differentiable training while encouraging sparse rule templates, explicit feature masking, signed evidence integration, and robust margin matching.

= Related Work

== Interpretable Tabular Models

Interpretable tabular learning includes sparse linear models (e.g., LASSO and elastic net @tibshirani1996 @zou2005), decision trees, rule lists @angelino2018learning, rule ensembles @wang2016bayesian @dash2018boolean, generalized additive models @hastie1990generalized, and explainable boosting machines @lou2013 @nori2019. These methods vary in how they optimize rules, capture multivariate interactions, and scale computationally. For a comprehensive review of deep tabular learning, see @borisov2022deep.

== Rule Ensembles

#RuleFit formulates predictions as sparse linear combinations of rules extracted heuristically from tree ensembles @friedman2008rulefit. This decouples rule generation from coefficient estimation. Conversely, #MTE jointly optimizes both the rule predicates and activation weights via continuous gradient descent.

== Additive Glassbox Models

Explainable Boosting Machines (#EBM) learn additive shape functions and selected pairwise interactions @lou2013 @nori2019. #EBM frequently achieves state-of-the-art performance on tabular datasets and serves as our primary glassbox baseline. #MTE provides a complementary approach: instead of learning univariate or pairwise additive functions, it learns sparse, conjunction-style templates over multiple continuous signed literals.

== Gradient-Boosted Trees

#XGBoost, LightGBM, and #CatBoost represent the state-of-the-art for tabular prediction @chen2016xgboost @ke2017lightgbm @prokhorenkova2018catboost @grinsztajn2022why. While they consistently yield high predictive accuracy, obtaining global explanations from these models typically requires post-hoc attribution or surrogate-analysis tools that do not yield the same compact rule template as the trained model itself.

== Differentiable Trees and Neural Rules

Deep tabular architectures such as NODE, Net-DNF, and TabNet incorporate structural tabular priors, including oblivious routing, disjunctive normal forms, and sequential attention @popov2020neural @katz2021net @arik2021tabnet. Similarly, differentiable decision trees and neural rule architectures replace hard routing thresholds with sigmoidal or softmax approximations @irsoy2012 @kontschieder2015 @evans2018learning @payani2019inductive. #MTE diverges from these approaches by representing rules using signed log-margin literals and quantifying rule activation through explicit, probabilistic energy templates rather than hierarchical tree routing.

== Robust Energies

The Student-$t$ distribution is widely used in robust statistics @huber1981robust @lange1989robust because its negative log-density scales logarithmically rather than quadratically with respect to residual magnitude. #SMTE leverages this property to introduce outlier-resistant template matching into signed log-margin rule optimization.

= Design Rationale

== Soft Axis-Aligned Literals

Axis-aligned threshold predicates are the basic building blocks of decision trees and rule ensembles. They offer high interpretability because each condition isolates a single feature and a discrete threshold. However, hard predicates introduce step-function discontinuities, making gradient-based optimization impossible. To resolve this, we use a continuous sigmoidal relaxation:

$ c_(r j)(x) = sigma(kappa d_(r j) (x_j - theta_(r j))). $ <eq-comparison-basic>

The threshold $theta_(r j)$, direction scalar $d_(r j)$, and gating mask $m_(r j)$ are jointly learned. The literal inverse temperature $kappa$ is represented by a learned scalar log-parameter $eta_kappa$, initialized as $eta_kappa = log 6.0$, and transformed during the forward pass as
$ kappa = min(50.0, max(0.5, exp(eta_kappa))). $
Thus, $kappa$ is optimized jointly with the rule parameters while remaining numerically bounded, dictating the sharpness of the soft comparison.

== Signed Evidence

Conventional soft rules treat literal satisfaction strictly as positive evidence. However, in complex tabular environments, crossing a threshold may either support or contradict a rule depending on the multivariate context. We resolve this by introducing an evidence-sign parameter $s_(r j) in (-1, 1)$, allowing each literal to contribute along a continuous spectrum of positive or negative evidence.

== Log-Margins

Given a pair $(u_(r j), v_(r j))$ representing positive and negative evidence, we summarize their relative magnitude using the log-ratio:

$ z_(r j)(x) = log(u_(r j)(x) + epsilon) - log(v_(r j)(x) + epsilon). $ <eq-log-margin-basic>

A large positive $z_(r j)$ indicates that positive evidence dominates, a large negative $z_(r j)$ indicates that negative evidence dominates, and values near zero signify ambiguity. As the feature mask $m_(r j) -> 0$, the evidence components $u_(r j), v_(r j) -> 0$, forcing the log-margin $z_(r j) -> 0$. This acts as a constant energy offset that can be absorbed by the baseline threshold $b_r$, thereby neutralizing the feature's influence on the rule activation.

== Probabilistic Templates

Rather than defining rule activation as a static Boolean conjunction, #MTE optimizes a probabilistic template in margin space. The location parameter $mu_(r j)$ defines the target margin value for feature $j$ within rule $r$, while the scale parameter $tau_(r j)$ defines the acceptable tolerance. Consequently, an #MTE rule is interpretable not only by its thresholds, but also by the target margin satisfaction required for activation.

== Student-$t$ Templates

Gaussian energy functions penalize margin residuals quadratically. This makes optimization sensitive to extreme margin values, which can be problematic because log-margins may become large when evidence is highly skewed. Student-$t$ energies mitigate this by enforcing logarithmic growth for large residuals, resulting in bounded score functions. This provides a theoretically robust, outlier-resistant template-matching mechanism in margin space.

= Margin-Template Rule Ensembles

Let $x in RR^D$ denote an input vector, $r in {1, dots, R}$ index the rules, and $j in {1, dots, D}$ index the features. For classification with $C$ classes, the network outputs logits in $RR^C$; for regression, it outputs a scalar prediction.

== Soft Signed Literals

Each rule-feature pair has a threshold $theta_(r j)$, inequality weight $a_(r j)$, evidence-sign weight $e_(r j)$, and mask logit $omega_(r j)$. We define the constrained parameters:

$ d_(r j) = tanh(a_(r j)) in (-1, 1), \
  s_(r j) = tanh(e_(r j)) in (-1, 1), \
  m_(r j) = sigma(omega_(r j)) in (0, 1). $ <eq-parameters>

Here, $d_(r j)$ controls the functional inequality direction, $s_(r j)$ dictates whether condition satisfaction contributes positive or negative evidence, and $m_(r j)$ acts as a sparsity-inducing feature selector.

In addition to the rule-feature parameters, the model contains a learned scalar literal inverse-temperature parameter $eta_kappa$. The operative inverse temperature is bounded to ensure numerical stability:

$ kappa = min(kappa_max, max(kappa_min, exp(eta_kappa))), quad kappa_min = 0.5, quad kappa_max = 50.0. $ <eq-kappa-param>

Given the learned inverse temperature $kappa$, the continuous comparison is:

$ c_(r j)(x) = sigma(kappa d_(r j) (x_j - theta_(r j))). $ <eq-comparison>

For $d_(r j) > 0$, this approximates $x_j > theta_(r j)$; for $d_(r j) < 0$, it approximates $x_j < theta_(r j)$. In the implementation, $eta_kappa$ is initialized with $eta_kappa = log 6.0$ and optimized jointly with the model. 

The positive and negative evidence components are defined as:

$ u_(r j)(x) = m_(r j)[ frac(1 + s_(r j), 2) c_(r j)(x) + frac(1 - s_(r j), 2) (1 - c_(r j)(x)) ], \
  v_(r j)(x) = m_(r j) - u_(r j)(x). $ <eq-evidence>

The resulting signed soft literal is:

$ ell_(r j)(x) = m_(r j) s_(r j) (2 c_(r j)(x) - 1). $ <eq-signed-literal>

== Signed Log-Margin Representation

For each rule-feature combination, the signed log-margin is computed as:

$ z_(r j)(x) = log(u_(r j)(x) + epsilon) - log(v_(r j)(x) + epsilon), $ <eq-logmargin>

where $epsilon > 0$ ensures numerical stability.

== Gaussian Margin Template Ensemble

The Gaussian template assigns rule $r$ a location $mu_(r j)$ and scale $tau_(r j) > 0$ in margin space. Template scales are parameterized through a softplus transform to ensure positivity. The associated energy is:

$ E^G_r(x) = - sum_(j=1)^D frac((z_(r j)(x) - mu_(r j))^2, 2 tau_(r j)^2). $ <eq-gmte-energy>

The rule activation function is:

$ h^G_r(x) = sigma(beta (E^G_r(x) - b_r)), $ <eq-gmte-activation>

where $beta > 0$ controls the rule-activation sharpness and $b_r$ is a learned baseline threshold. The rule-activation sharpness $beta$, by contrast to $kappa$, is treated as a fixed global hyperparameter.

== Student-$t$ Margin Template Ensemble

The Student-$t$ template incorporates a location $mu_(r j)$, scale $tau_(r j) > 0$, and a degrees-of-freedom parameter $nu_r > 0$. For #SMTE, the degrees-of-freedom parameter is represented through a softplus transform and clamped to the interval $[1, 50]$ in the implementation. Its energy is:

$ E^T_r(x) = - sum_(j=1)^D frac(nu_r + 1, 2) log(1 + frac((z_(r j)(x) - mu_(r j))^2, nu_r tau_(r j)^2)). $ <eq-smte-energy>

The rule activation function is:

$ h^T_r(x) = sigma(beta (E^T_r(x) - b_r)). $ <eq-smte-activation>

Predictions are generated via an affine projection over rule activations:

$ f(x) = W h(x) + q, quad h(x) = (h_1(x), dots, h_R(x))^top. $ <eq-head>

For regression, $W in RR^(1 times R)$ and $q in RR$. For classification, $W in RR^(C times R)$ and $q in RR^C$.

== Training Objective

Let $ell_("sup")$ denote the supervised loss. For regression, $ell_("sup")$ is mean squared error on standardized targets:
$ ell_("sup")(f_Theta(x_i), y_i) = (f_Theta(x_i) - y_i)^2. $ <eq-reg-loss>

For classification, $ell_("sup")$ is multiclass cross-entropy applied to logits:
$ ell_("sup")(f_Theta(x_i), y_i)
= - log frac(exp(f_Theta(x_i)_(y_i)), sum_(c=1)^C exp(f_Theta(x_i)_c)). $ <eq-cls-loss>

The training objective combines the supervised loss with an explicit sparsity penalty on the continuous masks:

$ cal(L)(Theta) = frac(1, n) sum_(i=1)^n ell_("sup")(f_Theta(x_i), y_i) + lambda_m sum_(r=1)^R sum_(j=1)^D m_(r j). $ <eq-objective>

We optimize @eq-objective using AdamW with decoupled weight decay $10^(-4)$ @loshchilov2019decoupled. The weight decay is optimizer-level regularization and is not written as an explicit $lambda_2 ||Theta||_2^2$ term in @eq-objective. We combine this with cosine annealing @loshchilov2017sgdr, gradient clipping, and validation-based early stopping. The complete optimization procedure is summarized in @tab-training.

#here-table(
  compact-table[
    #table(
      columns: (2.2em, 1fr),
      inset: 3pt,
      align: horizon,
      table.header([*Step*], [*Operation*]),
      [1], [Initialize thresholds $theta_(r j)$ using empirical feature quantiles.],
      [2], [Initialize mask logits with negative values to encourage initial sparsity.],
      [3], [Initialize template locations, scales, tail parameters, and affine head weights.],
      [4], [For each minibatch, compute continuous comparisons, evidence parameters, log-margins, and template energies.],
      [5], [Compute rule activations and predictions, and evaluate the loss in @eq-objective.],
      [6], [Update parameters via AdamW with gradient clipping.],
      [7], [Retain the best validation checkpoint and apply early stopping upon convergence.],
    )
  ],
  [Optimization procedure for Margin-Template Rule Ensembles.]
) <tab-training>

= Rule Extraction and Interpretation

Following optimization, the #MTE architecture can be extracted into a ranked set of human-readable rule templates. This extraction uses four learned quantities:

1. The head-weight magnitude, representing the predictive influence of rule $r$;
2. The mask activation $m_(r j)$, indicating whether feature $j$ is active in rule $r$;
3. The direction parameter $d_(r j)$, showing whether the inequality is $x_j > theta_(r j)$ or $x_j < theta_(r j)$;
4. The evidence sign $s_(r j)$, indicating whether satisfying the condition provides positive or negative evidence for rule activation.

== Displayed Conditions

In the experiments, displayed literals are selected using a rule-specific cutoff rather than a single global threshold. For rule $r$, define

$ gamma_r = max(gamma_("abs"), gamma_("rel") max_j m_(r j)). $ <eq-dynamic-display-cutoff>

The active displayed literals are then

$ A_r(gamma_r) = {j : m_(r j) >= gamma_r}. $ <eq-displayed-set>

For each $j in A_r(gamma_r)$, the human-readable condition is $x_j > theta_(r j)$ if $d_(r j) > 0$, and $x_j < theta_(r j)$ if $d_(r j) < 0$. The evidence direction is determined by the sign of $s_(r j)$. In the synthetic benchmark, we use $gamma_("abs") = 0.05$ and $gamma_("rel") = 0.20$. For rule examples in the extended tabular benchmark, we use $gamma_("abs") = 0.01$ and $gamma_("rel") = 0.25$.

== Rule Importance

In regression, the importance of rule $r$ is the absolute head weight:

$ cal(I)_r = |W_r|. $ <eq-reg-rule-importance>

For multiclass classification, where $W in RR^(C times R)$, the importance is the $L_1$ norm of the class-specific head weights:

$ cal(I)_r = ||W_(dot r)||_1. $ <eq-class-rule-importance>

This matches the extraction procedure used in the experiments, where rule importance is computed as the sum of absolute head weights across output dimensions. Extracted rules are ranked in descending order of $cal(I)_r$ to construct a global model summary.

== Rule Sparsity and Mask Entropy

The structural complexity of rule $r$ is the number of active displayed literals:

$ C_r(gamma_r) = |A_r(gamma_r)|. $ <eq-rule-complexity>

The mean structural complexity across the top $K_("show")$ rules is:

$ overline(C) = frac(1, K_("show")) sum_(r in cal(R)_("top")) C_r(gamma_r). $ <eq-avg-complexity>

We define the normalized rule sparsity as:

$ S_("rule") = 1 - frac(overline(C), D). $ <eq-rule-sparsity>

To measure the definiteness of the continuous masks, we compute the average binary cross-entropy of the mask distribution:

$ H_("mask") = - frac(1, R D) sum_(r=1)^R sum_(j=1)^D \[ m_(r j) log(m_(r j) + epsilon) + (1 - m_(r j)) log(1 - m_(r j) + epsilon) ]. $ <eq-mask-entropy>

A lower $H_("mask")$ indicates that the masks are closer to binary states, leading to sharper structural boundaries.

= Mathematical Properties

This section presents the formal theoretical properties of #MTE. Detailed proofs follow standard literature in smooth functional analysis, approximation theory, robust statistics, and empirical process theory.

== Standing Assumptions

We define the constants

$ M_z := log(frac(1 + epsilon, epsilon)), quad L_c := frac(kappa, 4). $ <eq-mz-lc>

Here, $M_z$ bounds the magnitude of the signed log-margin, and $L_c$ is the maximal derivative of the sigmoid comparison with respect to its scalar input, scaled by the literal inverse temperature $kappa$. Since $kappa$ is bounded in $[0.5, 50.0]$ during optimization, $L_c$ remains strictly bounded.

#assumption("Bounded Parameter Set",[
There exist positive constants $B_Theta, B_W, B_q, B_x, tau_min, nu_min > 0$ such that $|theta_(r j)|, |mu_(r j)| <= B_Theta$, $tau_(r j) >= tau_min$, $nu_r >= nu_min$, $||W||_F <= B_W$, and $||q||_2 <= B_q$ (used in the output bound $B_f$). Furthermore, the input domain is bounded as $cal(X) subset.eq [-B_x, B_x]^D$, and $epsilon in (0, 1 / 2]$ is fixed.
])

== Smoothness, Range, and Complexity

#proposition("Smoothness",[
Under the bounded parameter assumption, both #SMTE and #GMTE yield functions $f_Theta : cal(X) -> RR^C$ that are $C^infinity$ smooth in $x$. They are $C^infinity$ smooth in $Theta$ almost everywhere, becoming piecewise smooth at the boundaries of the $kappa$ clamp, conditional on fixed $beta$ and $epsilon > 0$.
])

#proposition("Computational Complexity",[
For $n$ input samples, $D$ features, $R$ rules, and $C$ prediction targets, a single forward pass requires $O(n R D + n R C)$ operations. The parameter space scales as $O(R D + R C)$, and the memory footprint of intermediate activations scales as $O(n R (D + C))$.
])

#proposition("Range of Log-Margins",[
For any $(x, r, j)$, $c_(r j)(x) in (0, 1)$, $u_(r j)(x), v_(r j)(x) in [0, m_(r j)] subset [0, 1]$, and $u_(r j)(x) + v_(r j)(x) = m_(r j)$. The log-margin is bounded by $|z_(r j)(x)| <= M_z = log((1 + epsilon) / epsilon)$, which yields bounded residuals $|z_(r j)(x) - mu_(r j)| <= M_z + B_Theta$.
])

#proposition("Soft-Literal Derivative Bound",[
For any bounded literal inverse temperature $kappa$, rule $r$, feature $j$, and input $x$, the partial derivatives are bounded by $|partial_(x_j) c_(r j)(x)| <= kappa / 4$ and $|partial_theta c_(r j)(x)| <= kappa / 4$. These bounds are approached asymptotically at the decision boundary $x_j = theta_(r j)$ as $|d_(r j)| -> 1$.
])

#theorem("Quantitative Input Lipschitz Constant",[
Under the bounded parameter assumption, every target projection $f_c$ of the #MTE network is Lipschitz continuous on $cal(X)$. For all $x, x' in cal(X)$,
$ |f_c(x) - f_c(x')| <= L_f ||x - x'||_2, $
with the explicit upper bound
$ L_f = frac(B_W beta kappa_max (1 + epsilon) sqrt(R D), 16 epsilon) dot K_("tmpl"). $
The template-sensitivity factor is
$ K^G_("tmpl") = frac(M_z + B_Theta, tau_min^2) quad ("Gaussian"), quad
  K^T_("tmpl") = frac(nu_min + 1, 2 tau_min sqrt(nu_min)) quad ("Student-"t). $
The bound is obtained by composing the per-stage tight Lipschitz factors: the rule-activation factor $beta / 4$ from $sup_t sigma'(t) = 1/4$ @goodfellow2016deep, the template-energy factor $K_("tmpl")$ from the Sharp Influence-Function Comparison, the log-margin factor $kappa(1+epsilon)/(4 epsilon)$ from the range of log-margins together with $u_(r j) + v_(r j) = m_(r j)$, the per-rule aggregation factor $sqrt(D)$ via Cauchy–Schwarz on the axis-aligned literals, and the cross-rule factor $sqrt(R)$ via Cauchy–Schwarz on the head projection.
])

_Note: Because $kappa$ is constrained to $[0.5, 50.0]$ throughout training, $L_f$ is uniformly bounded irrespective of the learned value of $eta_kappa$. The Student-$t$ factor $K^T_("tmpl")$ is independent of the residual magnitude, whereas $K^G_("tmpl")$ scales with $M_z + B_Theta$. This establishes the formal robustness advantage of #SMTE._

== Hard-Rule Limit and Approximation

#theorem("Quantitative Hard-Rule Convergence",[
Fix $x in RR^D$ and rule $r$. Assume that for all $j$, the parameters are discrete: $|d_(r j)| = delta in (0, 1]$, $m_(r j) in {0, 1}$, $s_(r j) in {-1, +1}$, and the inputs are bounded away from the decision boundaries: $|x_j - theta_(r j)| >= Delta_j > 0$. Setting $Delta = min_j Delta_j$, the continuous comparison converges to a strict Heaviside step function $c^star_(r j)(x)$ at an exponential rate:
$ |c_(r j)(x) - c^star_(r j)(x)| <= exp(-kappa delta Delta_j). $
Consequently, log-margins and rule activations converge uniformly on compact subsets interior to the threshold hyperplanes.
])

#theorem("Quantitative Universal Approximation", [
Let $f^star : [0, 1]^D -> RR$ be an $L$-Lipschitz target function. For any integer $N >= 1$, there exists an #MTE configuration with $R = N^D$ rules such that:
$ sup_(x in [0,1]^D) |f(x) - f^star(x)| <= frac(L sqrt(D), N) + L sqrt(D) rho(kappa, beta), $
where the relaxation error $rho(kappa, beta) -> 0$ as $kappa, beta -> infinity$. Selecting $kappa = 4 N log N$ and $beta = log N$ yields the uniform bound:
$ sup_x |f(x) - f^star(x)| <= 2 L sqrt(D) R^(-1 / D). $
])

_Note: The value $kappa = 4 N log N$ in this theorem refers to the theoretical capacity of the architecture. In the practical implementation, $kappa$ is numerically bounded to $[0.5, 50.0]$, meaning this approximation rate describes an asymptotic capacity bound rather than a directly achievable operational setting for arbitrarily large $N$._

While the theoretical bound $O(R^{-1/D})$ degrades in high dimensions, this theorem establishes baseline approximation capacity. The requirement of $R = N^D$ rules scales exponentially with dimensionality, suggesting that the empirical success of rule ensembles is driven by structural inductive bias and optimization rather than dense grid approximation.

#theorem("Minimax Lower Bound",[
Let $Lambda_L$ represent the class of $L$-Lipschitz functions over $[0,1]^D$. Any rule-parameterized architecture with pseudo-dimension $O(R polylog R)$ incurs a worst-case uniform approximation error of at least $c L R^(-1 / D)$ for some constant $c > 0$. Thus, the #MTE approximation rate is minimax-optimal up to logarithmic factors and constants @devore1993constructive @yang1999information.
])

== Robustness of Student-$t$ Templates

#theorem("Sharp Influence-Function Comparison",[
Following the classical influence-function formalism of robust statistics @hampel1974influence @huber1981robust, define the loss energies $rho_T(xi) := (nu + 1) / 2 dot log(1 + xi^2 / (nu tau^2))$ and $rho_G(xi) := xi^2 / (2 tau^2)$. The maximal influence for the Student-$t$ energy is bounded:
$ sup_xi |rho'_T(xi)| = frac(nu + 1, 2 tau sqrt(nu)), $
occurring at $xi = plus.minus tau sqrt(nu)$. Furthermore, $rho'_T(xi) -> 0$ as $|xi| -> infinity$, indicating a redescending score function. Conversely, the Gaussian influence $rho'_G(xi) = xi / tau^2$ grows without bound.
])

#corollary("Gross-Error Sensitivity",[
Within #SMTE, the per-feature gradient contribution of $E^T_r$ with respect to $z_(r j)$ is strictly bounded by $(nu_r + 1) / (2 tau_(r j) sqrt(nu_r))$. Consequently, the gross-error sensitivity is finite for #SMTE, but infinite for #GMTE.
])

#proposition("Gaussian Limit of Student-$t$",[
For a fixed residual $xi$ and scale parameter $tau$, Taylor expansion yields:
$-(nu + 1) / 2 dot log(1 + xi^2 / (nu tau^2)) = -xi^2 / (2 tau^2) + O(1 / nu)$
as $nu -> infinity$. Thus, #GMTE represents the asymptotic limit of #SMTE as the degrees of freedom approach infinity.
])

== Generalization and Recovery Diagnostics

#theorem("Rademacher Complexity",[
Let $cal(F)_(R,D)$ denote the hypothesis class of scalar #MTE functions adhering to the bounded parameter assumptions. With $P = O(R D)$ effective parameters, a uniform output bound $B_f$, and the Lipschitz constant $L_f$, standard covering-number machinery @anthony1999neural @bartlett2002rademacher together with Dudley's entropy integral @dudley1967sizes yield, for some universal constant $C_1 > 0$:
$ frak(R)_n(cal(F)_(R,D)) <= frac(C_1 B_f, sqrt(n)) dot sqrt(P log(1 + frac(n L_f^2, P))) = O(sqrt(frac(P log n, n))). $
])

#theorem("Finite-Sample Complexity Bound",[
Let $cal(E)(f)$ denote the true risk and $hat(cal(E))_n(f)$ the empirical risk of $f$. Assuming an $L_ell$-Lipschitz supervised loss function bounded by $M$, then by standard Rademacher-based generalization bounds and bounded differences @mohri2018foundations @bartlett2002rademacher @mcdiarmid1989, with probability at least $1 - delta$, any empirical minimizer $f in cal(F)_(R,D)$ satisfies the generalization bound:
$ cal(E)(f) <= hat(cal(E))_n(f) + O(L_ell sqrt(frac(P log n, n)) + M sqrt(frac(log(1 / delta), n))). $
])

To estimate feature importance within a fully differentiable framework, we calculate the average absolute input gradient:

$ hat(S)_j = frac(1, n) sum_(i=1)^n |frac(partial f(x_i), partial x_j)|. $ <eq-saliency>

Interactions are quantified via the absolute empirical covariance of the input gradients:

$ hat(I)_(j k) = | frac(1, n) sum_(i=1)^n (g_(i j) - overline(g)_j)(g_(i k) - overline(g)_k) |, quad g_(i j) = frac(partial f(x_i), partial x_j). $ <eq-interaction>

#theorem("Active-Feature Recovery",[
Assume population saliencies satisfy $S_j >= a$ for active features and $S_j <= b$ for null features, where $a > b$. If gradients are bounded by $M$, then by Hoeffding's inequality @hoeffding1963, selecting the top $|A|$ empirical saliency scores isolates the active feature set with probability at least:
$ 1 - 2 D exp(- frac(n (a - b)^2, 2 M^2)). $
])

#theorem("Interaction-Pair Recovery",[
Let $g_j(X) := partial f(X) / partial x_j$ and assume $|g_j(X)| <= M$ almost surely for every feature $j$. Define the population interaction score
$ I_(j k) := | EE[g_j(X) g_k(X)] |, $
and its empirical counterpart
$ hat(I)_(j k) := | frac(1, n) sum_(i=1)^n g_(i j) g_(i k) |. $
Assume the population scores satisfy a margin condition: $I_(j k) >= a_I$ for true interaction pairs and $I_(j k) <= b_I$ for null pairs, with $a_I > b_I$. Then, by Hoeffding's inequality and concentration of bounded differences @hoeffding1963 @mcdiarmid1989 @boucheron2013concentration applied to the products $g_(i j) g_(i k) in [-M^2, M^2]$ and a union bound over the $D^2$ candidate pairs, retrieving the top subset of empirical interaction scores recovers the true interaction set with probability at least:
$ 1 - 2 D^2 exp(- frac(n (a_I - b_I)^2, 8 M^4)). $
])

_Note: This theorem uses the uncentered second moment $EE[g_j g_k]$, which equals $op("Cov")(g_j, g_k)$ when the gradients are centered ($EE[g_j] = 0$)---a condition that holds in expectation at unconstrained optima. If a centered estimator is used directly, the centered products lie in $[-4M^2, 4M^2]$, and the analogous bound holds with denominator $128M^4$; this only weakens the constant, leaving the exponential rate $Theta(n \/ M^4)$ unchanged._

= Experimental Setup

== Evaluated Models

We benchmark #SMTE and #GMTE against four tabular baselines: #EBM, #XGBoost, #CatBoost, and #RuleFit.

== Synthetic Evaluation

The synthetic benchmark contains five controlled datasets with mathematically defined active features and interactions:

1. *Hierarchical*: A threshold rule combined with gated linear terms and interactions.
2. *Friedman1*: A complex nonlinear regression problem dependent on five active features.
3. *XorChain*: A multiclass problem requiring XOR and logical conjunction learning.
4. *ThresholdCls*: Binary classification driven by a combination of threshold and linear scores.
5. *MultiThresh*: Multiclass threshold conjunctions.

Each dataset contains $n = 2000$ samples and $D = 15$ dimensions. All experiments are conducted across five random seeds: 1, 12, 123, 1234, and 12345.

== Empirical Evaluation on Tabular Benchmarks

The extended benchmark uses 28 tabular datasets, consisting of 25 real-world datasets and 3 synthetic regression generators: Friedman1--3. The Boston dataset is retained to evaluate historical baseline performance alongside modern benchmarks such as California Housing.

*Regression:* California, Diabetes, Friedman1, Friedman2, Friedman3, Abalone, Concrete, Energy, Yacht, Kin8nm, Boston, CPU_Act, and Auto_MPG.

*Classification:* Wine, Iris, Digits, BreastCancer, CovType, Spambase, Ionosphere, Sonar, Banknote, Beans, HeartStatlog, Adult, GermanCredit, Vehicle, and Mushroom.

Each dataset is evaluated across the same five random seeds. For each seed, the data is partitioned into train, validation, and test splits of 64% / 16% / 20%. Numeric columns are median-imputed and standardized. Categorical variables are mode-imputed and one-hot encoded. Regression target variables are normalized during optimization and inverse-transformed for evaluation.

== Implementation Details and Baselines

For the synthetic benchmark, #MTE architectures use $R = 32$ rules, 150 epochs, early stopping patience of 20, batch size 256, learning rate $3 times 10^(-3)$, AdamW weight decay $10^(-4)$, and a mask penalty $10^(-4)$. Baseline #XGBoost uses 300 trees of depth 5; #CatBoost uses 400 iterations of depth 5; and #RuleFit uses 200 estimators limited to 2000 rules. These specific hyperparameter assignments accommodate the lower dimensionality and specific interaction scales of the synthetic environments.

For the tabular benchmark, #MTE uses $R = 64$ rules, 200 epochs, patience 40, batch size 512, with the same learning rate, weight decay, and mask penalty. Baseline #XGBoost uses 400 trees of depth 4; #CatBoost uses 500 iterations of depth 5; and #RuleFit uses 200 estimators limited to 300 rules. In all experiments, #EBM is configured with 10 interactions, and tree-based ensembles use a learning rate of 0.05 (0.04 for #CatBoost) with standard subsampling.

In the #MTE implementation, the literal inverse temperature $kappa$ is learned as a global scalar through a bounded exponential parameterization. Specifically, the model stores $eta_kappa$ and computes
$ kappa = min(50.0, max(0.5, exp(eta_kappa))), $
with an initialization of $eta_kappa = log 6.0$. The rule-activation sharpness $beta$ is fixed to $6.0$ in all reported experiments. Template scales are constrained to be positive through a softplus transform. For #SMTE, the degrees-of-freedom parameter $nu_r$ is parameterized through a softplus transform and clamped to $[1, 50]$.

== Evaluation Metrics and Statistical Ranking

Regression tasks are evaluated using RMSE; classification tasks use macro-F1. For each dataset-seed block, models are ranked by performance. A normalized score assigns 1.0 to the best model and 0.0 to the weakest within each block.

For synthetic tasks, ground-truth active features are known, allowing us to evaluate feature recovery, interaction detection, structural sparsity, and directional alignment. These are combined into a composite interpretability score:

$ I = 0.40 I_("feat") + 0.25 I_("sparse") + 0.20 I_("interact") + 0.15 I_("direction"). $ <eq-interpretability-score>

To evaluate statistical significance across the extended benchmark, the Nemenyi post-hoc test critical difference at significance level $alpha = 0.05$ @demsar2006statistical is given by:

$ "CD" = q_alpha sqrt(frac(M(M + 1), 6 N)). $ <eq-cd>

In our extended tabular benchmark ($M = 6$, $N = 140$ evaluation blocks), the block-wise critical difference is approximately 0.64.

== Comprehensive Ablation Study

To isolate the impact of individual architectural mechanisms, we conduct an ablation and sensitivity study using the five synthetic datasets across five seeds.

The reference configuration is standard #SMTE with $R = 32$, 150 epochs, patience 20, batch size 256, learning rate $3 times 10^(-3)$, AdamW weight decay $10^(-4)$, mask penalty $lambda_m = 10^(-4)$. The evaluated configurations include:

1. *Reference #SMTE*: The complete Student-$t$ signed log-margin architecture.
2. *No Signed Evidence*: Restricting $s_(r j) = 1$, which forces condition satisfaction to contribute only positive evidence.
3. *No Log-Margin*: Substituting the signed log-margin $z_(r j)$ with the raw signed literal $ell_(r j)$.
4. *No Mask Penalty*: Setting $lambda_m = 0$ to evaluate the effect of sparsity regularization.
5. *Fixed Thresholds*: Initializing thresholds via empirical quantiles but freezing them during optimization.
6. *Fixed Scales*: Fixing $tau_(r j) = 1$ to disable adaptive template tolerances.
7. *Gaussian Template*: Substituting the Student-$t$ formulation with a Gaussian template.

We also evaluate structural sensitivity by varying rule capacity ($R in {16, 32, 64, 128}$) and mask penalty strength ($lambda_m in {0, 10^(-5), 10^(-4), 10^(-3)}$).

The ablation suite consists of 5 datasets, 5 seeds, and 15 configurations, totaling 375 model runs. Performance is normalized per dataset-seed ablation block, and interpretability is measured using @eq-interpretability-score. Ablation-normalized performance scores are therefore not directly comparable to the benchmark-normalized scores in @tab-synthetic-summary, which include external baselines.

= Empirical Findings

The experimental results yield the following statistical observations:

1. *#MTE recovers structural dependencies.* Under ground-truth conditions, #GMTE achieves a feature-selection F1 score of 1.000, while #SMTE achieves 0.990. Both variants accurately identify active features and outperform heuristic extraction baselines in interaction recovery.
2. *#GMTE favors low-noise interpretability; #SMTE favors empirical performance.* #GMTE yields the highest composite interpretability on synthetic tasks, whereas #SMTE secures a superior rank among the proposed models on the extended tabular benchmark.
3. *#SMTE matches the performance of standard baselines.* Across 140 extended benchmark evaluation blocks, #SMTE achieves an average rank of 3.104, trailing #EBM (3.021) and #CatBoost (3.046) by margins well within the critical difference threshold of 0.64.
4. *Log-margins and bidirectional evidence drive performance.* Disabling the log-margin logic drops ablation-normalized performance from 0.824 to 0.386. Forcing strict positive evidence reduces performance to 0.519.
5. *The mask penalty controls rule sparsity.* Removing the mask penalty increases mask entropy from 0.199 to 0.367 and decreases structural sparsity from 0.819 to 0.733, with only a marginal predictive gain (0.877).

== Synthetic Evaluation

@tab-synthetic-summary summarizes the model performance over the 25 synthetic runs. Model rank measures predictive capability, while the composite score $I$ measures interpretability against the ground truth. #GMTE achieves the highest composite interpretability score, closely followed by #SMTE. Both #MTE variants achieve high recovery of active features: #GMTE yields a feature-selection F1 of 1.000 ± 0.000, and #SMTE yields 0.990 ± 0.050. In this implementation, #RuleFit feature recovery is scored as 0 because the extracted rule representation did not yield reliable feature-importance matches under the common parsing procedure. Both #MTE variants outperform the tree-based baselines in interaction recovery, validating the use of the differentiable gradient-covariance method.

#wide-table(
  compact-table[
    #table(
      columns: (1fr, 1fr, .65fr, .75fr, .7fr, .7fr, .7fr),
      inset: 2.5pt,
      align: center + horizon,
      table.header([*Model*], [*Perf.*], [*Rank*], [*$I$*],
        [*F1*], [*Inter.*], [*Time (s)*],
      ),
      [EBM], [0.909 ± 0.118], [2.30], [0.648], [0.870], [0.500], [65.97],
      [CatBoost], [0.894 ± 0.162], [2.44], [0.575], [1.000], [0.500], [1.01],
      [XGBoost], [0.609 ± 0.362], [3.44], [0.575], [1.000], [0.500], [0.47],
      [GMTE], [0.388 ± 0.388], [4.68], [*0.886*], [*1.000*], [*0.889*], [2.98],
      [SMTE], [0.365 ± 0.371], [4.44], [0.865], [0.990], [0.796], [3.59],
      [RuleFit], [0.527 ± 0.410], [3.70], [0.300], [0.000], [0.500], [72.73],
    )
  ],
  [Synthetic benchmark summary over 25 runs. Performance is dataset-normalized; interpretability is the composite score defined in @eq-interpretability-score. Training times represent hardware-specific profiles and are subject to minor execution environment variances.]
) <tab-synthetic-summary>

#here-figure(
  paper-image(
    "outputs_synthetic_paper/figures/figure_1_accuracy_interpretability_frontier.png",
    width: 90%,
  ),
  [Synthetic accuracy-interpretability frontier. The proposed #MTE frameworks occupy the high-interpretability region, while #EBM and #CatBoost achieve slightly stronger predictive scores on these low-dimensional synthetic tasks.]
) <fig-synthetic-frontier>

#here-figure(
  grid(
    columns: (1fr, 1fr),
    gutter: 6pt,
    paper-image(
      "outputs_synthetic_paper/figures/figure_3_interpretability_decomposition.png",
      width: 100%,
    ),
    paper-image(
      "outputs_synthetic_paper/figures/figure_4_feature_recovery_heatmap.png",
      width: 100%,
    ),
  ),
  [Decomposition of synthetic interpretability metrics (left) and feature-recovery heatmap across tasks (right).]
) <fig-synthetic-interpretability>

== Evaluation on Tabular Benchmarks

@tab-real-global presents the aggregate statistics across the 140 dataset-seed evaluation blocks (840 total runs). #EBM achieves the lowest average rank, followed by #CatBoost and #SMTE. Among the proposed architectures, #SMTE achieves the best performance, with an average rank of 3.104 and a normalized score of 0.670. #XGBoost is the fastest to optimize, while #RuleFit exhibits the longest training times.

#wide-table(
  compact-table[
    #table(
      columns: (1fr, .75fr, .7fr, .65fr, .7fr, .7fr, .55fr),
      inset: 2.5pt,
      align: center + horizon,
      table.header([*Model*], [*Group*], [*Rank*], [*Med.*],
        [*Perf.*], [*Time (s)*], [*Runs*],
      ),
      [EBM], [Base.], [*3.021*], [3.000], [0.670], [45.64], [140],
      [CatBoost], [Base.], [3.046], [3.000], [*0.736*], [0.96], [140],
      [SMTE], [Prop.], [3.104], [3.000], [0.670], [3.67], [140],
      [XGBoost], [Base.], [3.414], [3.500], [0.624], [*0.41*], [140],
      [GMTE], [Prop.], [3.554], [3.500], [0.576], [3.38], [140],
      [RuleFit], [Base.], [4.861], [5.750], [0.255], [157.02], [140],
    )
  ],
  [Global tabular benchmark summary encompassing 28 tabular datasets: 25 real-world datasets and 3 synthetic regression generators. Lower average rank indicates superior performance. Timing values are hardware-specific.]
) <tab-real-global>

#wide-table(
  compact-table[
    #table(
      columns: (.9fr, .9fr, .65fr, .65fr, .7fr, .5fr),
      inset: 2.5pt,
      align: center + horizon,
      table.header(
        [*Task*], [*Model*], [*Group*], [*Rank*], [*Perf.*], [*Runs*],
      ),
      [Class.], [CatBoost], [Base.], [*3.033*], [0.684], [75],
      [Class.], [SMTE], [Prop.], [3.287], [0.598], [75],
      [Class.], [EBM], [Base.], [3.293], [0.600], [75],
      [Class.], [XGBoost], [Base.], [3.360], [0.588], [75],
      [Class.], [GMTE], [Prop.], [3.527], [0.533], [75],
      [Class.], [RuleFit], [Base.], [4.500], [0.317], [75],
      [Reg.], [EBM], [Base.], [*2.708*], [0.750], [65],
      [Reg.], [SMTE], [Prop.], [2.892], [0.753], [65],
      [Reg.], [CatBoost], [Base.], [3.062], [*0.796*], [65],
      [Reg.], [XGBoost], [Base.], [3.477], [0.666], [65],
      [Reg.], [GMTE], [Prop.], [3.585], [0.627], [65],
      [Reg.], [RuleFit], [Base.], [5.277], [0.183], [65],
    )
  ],
  [Task-wise separation of average ranks and normalized performance metrics.]
) <tab-task-summary>

#wide-figure(
  paper-image("outputs_extended/figure_1_average_rank_cd.png", width: 92%),
  [Average-rank visualization incorporating the block-wise Nemenyi critical difference bounds (~0.64 across 140 evaluation blocks and 6 target models).]
) <fig-real-cd>

#wide-figure(
  paper-image(
    "outputs_extended/figure_2_dataset_performance_heatmap.png",
    width: 100%,
  ),
  [Dataset-level normalized predictive performance across all 28 tabular benchmark datasets.]
) <fig-real-heatmap>

#wide-figure(
  grid(
    columns: (1fr, 1fr),
    gutter: 6pt,
    paper-image(
      "outputs_extended/figure_3_efficiency_frontier.png",
      width: 100%,
    ),
    paper-image(
      "outputs_extended/figure_4_pairwise_win_matrix.png",
      width: 100%,
    ),
  ),
  [Extended tabular benchmark efficiency frontier (left) and pairwise win matrix indicating relative model dominance (right).]
) <fig-real-efficiency>

== Specific Task Performance

At the individual dataset level, #SMTE achieves the highest predictive performance on GermanCredit, Abalone, Friedman1, Friedman2, and Kin8nm. #GMTE is the average-rank winner on BreastCancer, Iris, and Wine. These results indicate that the structured, margin-template architectures are competitive with standard baselines while offering direct, continuous rule-level interpretability. Performance patterns show that the #MTE variants are highly effective in regression tasks, while #CatBoost maintains a stronger position in classification tasks.

== Learned Rule Structure

The #MTE framework constructs interpretable rules using learned literal masks, template locations, variance scales, evidence signs, tail-robustness parameters, and prediction head weights.

#wide-figure(
  paper-image(
    "outputs_synthetic_paper/figures/figure_6_rule_structure_Hierarchical.png",
    width: 100%,
  ),
  [Visualization of the learned #SMTE and #GMTE margin-rule configurations extracted from the Hierarchical synthetic dataset.]
) <fig-structure-hierarchical>

== Ablation and Sensitivity Analysis

The ablation framework evaluates the impact of each core component of the #MTE architecture. Performance scores in this subsection are normalized within each dataset-seed ablation block and should not be compared directly with @tab-synthetic-summary, which normalizes performance across the full benchmark model set including external baselines.

@tab-ablation-architecture details the core architectural permutations. The highest normalized performance score (0.877) is achieved by disabling the mask penalty entirely. However, this gain is accompanied by a decrease in rule readability: structural sparsity drops from 0.819 (Reference #SMTE) to 0.733, and mask entropy increases from 0.199 to 0.367. This shows that the mask penalty functions primarily as a regularization mechanism to ensure structural readability.

The signed log-margin representation is the most critical component for predictive performance. Substituting the signed log-margin with the raw signed literal (No log-margin) reduces normalized performance from 0.824 to 0.386, which is the largest performance drop among all configurations. This indicates that matching rules in a normalized evidence-ratio space is more effective than using basic continuous literals. Disabling bidirectional evidence (No signed evidence) decreases performance to 0.519, showing that allowing literals to provide either positive or negative evidence is key for mapping complex logical structures.

The Gaussian-template permutation performs similarly to the Student-$t$ reference on low-noise synthetic tasks (yielding a performance of 0.827 and composite interpretability of 0.881). 

Freezing thresholds (Fixed thresholds) and freezing scales (Fixed scales) reduce performance to 0.726 and 0.733, respectively, showing that both dynamic decision boundaries and adaptive template tolerances contribute to model fitness.

#wide-table(
  xcompact-table[
    #table(
      columns: (1.5fr, .5fr, .9fr, .8fr, .85fr, .75fr, .7fr, .7fr, .4fr),
      inset: 2pt,
      align: center + horizon,
      table.header(
        [*Ablation*], [*T.*], [*Perf.*], [*Rank*], [*Interp.*],
        [*F1*], [*Sparse*], [*Entr.*], [*n*],
      ),
      [No mask pen.], [S], [*0.877 ± 0.152*], [*2.52 ± 1.21*], [0.875 ± 0.047], [1.000 ± 0.000], [0.733 ± 0.000], [0.367 ± 0.001], [25],
      [Gaussian tmpl.], [G], [0.827 ± 0.191], [3.26 ± 1.47], [*0.881 ± 0.077*], [0.980 ± 0.100], [*0.843 ± 0.028*], [0.199 ± 0.001], [25],
      [Reference SMTE], [S], [0.824 ± 0.184], [3.28 ± 1.63], [*0.881 ± 0.068*], [*1.000 ± 0.000*], [0.819 ± 0.042], [0.199 ± 0.001], [25],
      [Fixed scales], [S], [0.733 ± 0.187], [4.50 ± 1.50], [0.875 ± 0.067], [0.990 ± 0.050], [0.827 ± 0.031], [0.199 ± 0.001], [25],
      [Fixed thresholds], [S], [0.726 ± 0.206], [4.62 ± 1.29], [0.859 ± 0.054], [1.000 ± 0.000], [0.810 ± 0.038], [0.200 ± 0.001], [25],
      [No signed evid.], [S], [0.519 ± 0.475], [4.80 ± 2.47], [0.831 ± 0.063], [0.990 ± 0.050], [0.779 ± 0.037], [0.201 ± 0.001], [25],
      [No log-margin], [S], [0.386 ± 0.459], [5.02 ± 2.52], [0.812 ± 0.130], [0.930 ± 0.170], [0.750 ± 0.020], [0.271 ± 0.014], [25],
    )
  ],
  [Architectural ablations evaluated across five synthetic datasets and five seeds (T: S = SMTE, G = GMTE). Lower rank indicates superior performance.]
) <tab-ablation-architecture>

#wide-figure(
  grid(
    columns: (1fr, 1fr),
    gutter: 6pt,
    paper-image(
      "outputs_ablation_paper/figures/figure_1_architecture_ablation_heatmap.png",
      width: 100%,
    ),
    paper-image(
      "outputs_ablation_paper/figures/figure_2_architecture_delta_lollipop.png",
      width: 100%,
    ),
  ),
  [Architecture ablation diagnostics. Left: Mean structural scores across permutations. Right: Normalized absolute performance variance relative to the Reference #SMTE baseline.]
) <fig-ablation-architecture>

@tab-ablation-rule-count shows the sensitivity of #SMTE to the total rule count ($R$). Increasing rule capacity improves performance from 0.229 at $R = 16$ to 0.745 at $R = 128$. Feature recovery remains high across all rule counts, while structural sparsity increases from 0.783 to 0.882. This indicates that scaling rule capacity improves function approximation without increasing the complexity of the extracted rules. Parameter counts scale linearly with $R$, matching the theoretical complexity of $O(R D + R C)$.

#wide-table(
  xcompact-table[
    #table(
      columns: (.4fr, .9fr, .85fr, .7fr, .75fr, .7fr, .7fr, .9fr, .4fr),
      inset: 2pt,
      align: center + horizon,
      table.header(
        [*$R$*], [*Perf.*], [*Interp.*], [*F1*], [*Sparse*],
        [*Entr.*], [*Time (s)*], [*Params*], [*n*],
      ),
      [16], [0.229 ± 0.389], [0.883 ± 0.052], [1.000 ± 0.000], [0.783 ± 0.040], [0.201 ± 0.001], [3.16 ± 0.07], [1510 ± 20], [25],
      [32], [0.621 ± 0.352], [0.883 ± 0.058], [1.000 ± 0.000], [0.816 ± 0.028], [0.199 ± 0.001], [3.16 ± 0.08], [3018 ± 39], [25],
      [64], [0.724 ± 0.356], [0.874 ± 0.058], [1.000 ± 0.000], [0.844 ± 0.032], [0.198 ± 0.000], [3.14 ± 0.07], [6032 ± 77], [25],
      [128], [*0.745 ± 0.360*], [*0.886 ± 0.039*], [1.000 ± 0.000], [*0.882 ± 0.038*], [*0.197 ± 0.002*], [3.13 ± 0.16], [12061 ± 154], [25],
    )
  ],
  [Sensitivity of structural approximation to the rule count ($R$) in #SMTE. Increasing rule capacity improves both performance and structural sparsity.]
) <tab-ablation-rule-count>

@tab-ablation-mask-penalty presents the sensitivity to the mask penalty ($lambda_m$). Increasing the mask penalty forces rules to utilize fewer features, lowering the mask entropy. Under a high penalty of $lambda_m = 10^(-3)$, structural sparsity reaches 0.975 and composite interpretability is 0.919. This regularization comes with a trade-off in predictive performance, as strong penalties restrict the active literals available for fine decision boundary fitting. The mask penalty encourages sparse rule templates, but the final number of displayed literals remains dataset-dependent, particularly after one-hot encoding of categorical variables.

#wide-table(
  xcompact-table[
    #table(
      columns: (.75fr, .9fr, .9fr, .75fr, .75fr, .75fr, .7fr, .4fr),
      inset: 2pt,
      align: center + horizon,
      table.header(
        [*$lambda_m$*], [*Perf.*], [*Interp.*], [*Sparse*],
        [*Cond.*], [*Entr.*], [*Time (s)*], [*n*],
      ),
      [$0$], [*0.553 ± 0.415*], [0.847 ± 0.070], [0.733 ± 0.000], [4.000 ± 0.000], [0.367 ± 0.001], [3.05 ± 0.07], [25],
      [$10^(-5)$], [0.488 ± 0.467], [0.858 ± 0.063], [0.734 ± 0.003], [3.992 ± 0.040], [0.218 ± 0.005], [3.16 ± 0.08], [25],
      [$10^(-4)$], [0.457 ± 0.423], [0.877 ± 0.062], [0.806 ± 0.042], [2.912 ± 0.627], [0.199 ± 0.001], [3.16 ± 0.07], [25],
      [$10^(-3)$], [0.527 ± 0.417], [*0.919 ± 0.061*], [*0.975 ± 0.026*], [*0.368 ± 0.394*], [*0.197 ± 0.000*], [3.14 ± 0.06], [25],
    )
  ],
  [Mask-penalty sensitivity within #SMTE. Higher regularization parameters reduce active literals and mask entropy, demonstrating the trade-off between accuracy and readability.]
) <tab-ablation-mask-penalty>

#wide-figure(
  grid(
    columns: (1fr, 1fr),
    gutter: 6pt,
    paper-image(
      "outputs_ablation_paper/figures/figure_4_rule_count_sensitivity.png",
      width: 100%,
    ),
    paper-image(
      "outputs_ablation_paper/figures/figure_5_mask_penalty_sensitivity.png",
      width: 100%,
    ),
  ),
  [Sensitivity trajectories for rule capacity (left) and mask regularization (right). Rule count controls model capacity, while the mask penalty controls structural density.]
) <fig-ablation-sensitivity>


= Limitations and Open Challenges

There are several areas for future work:

- *Axis-aligned constraints:* Continuous literals in #MTE are axis-aligned. Mapping rotational or oblique decision boundaries may require learning a larger number of rules. Incorporating learned geometric projections or coordinate rotations prior to literal evaluation is a potential area of improvement.
- *High-dimensional categorical spaces:* Standard one-hot encoding for high-cardinality categorical features can lead to a large feature dimension, resulting in redundant literals. Incorporating group-sparse penalties (e.g., group LASSO) on categorical blocks could address this.
- *Extraction thresholds:* The extraction of discrete rules relies on display cutoff parameters $(gamma_("abs"), gamma_("rel"))$ to convert continuous masks into displayed literals, and the number of active literals shown depends directly on this thresholding rule.
- *Non-convex optimization:* As a differentiable network, #MTE remains sensitive to initializations and temperature settings ($eta_kappa$). Poor initialization or excessively high initial temperatures can lead to vanishing gradients.
- *Interpretability validation:* On real-world datasets, the true underlying functions are unknown, meaning the extracted templates represent global model approximations rather than causal relationships.
- *Benchmark scope:* The ablation analysis was conducted primarily within controlled synthetic environments to isolate structural elements; sensitivity patterns may vary across empirical datasets with different noise characteristics.

= Discussion

The empirical evaluations show a clear split in model behavior. On controlled synthetic datasets, both #SMTE and #GMTE identify active features and recover high-order interactions, establishing a clear baseline for template-based rule extraction. On the extended tabular benchmark, #SMTE confirms its design rationale, proving competitive with existing glassbox frameworks (like #EBM) and ensemble methods (#CatBoost), particularly in regression settings. #GMTE provides strong synthetic transparency but exhibits slightly less stability on noisy, real-world datasets.

The ablation study demonstrates that using signed log-margins is essential to the model's predictive capacity. Bidirectional signed evidence is also necessary to capture complex contextual conditions. Allowing dynamic, gradient-based adjustments to thresholds and template scales improves function approximation, while the mask penalty acts as the primary parameter controlling rule sparsity and mask entropy.

#MTE does not aim to replace gradient-boosted trees for all tabular problems. Instead, it provides a differentiable, end-to-end rule template architecture with feature-recovery capabilities and Student-$t$ margin robustness. The Student-$t$ formulation bounds the influence of outlier margins, mitigating the optimization instabilities often associated with raw, skewed evidence margins.

Future research directions include investigating oblique threshold boundaries, enforcing monotonicity constraints, calibrating probabilistic predictions, exploring Bayesian priors for template parameters, and deploying hybrid models that combine additive shape functions with margin-template rule ensembles.

= Conclusion

This work introduces Margin-Template Rule Ensembles, a differentiable, interpretable tabular architecture. #MTE structures decision logic using soft axis-aligned literals, bidirectional evidence, continuous sparsity masks, and log-margin templates. While the Gaussian variant utilizes standard quadratic error margins, the Student-$t$ formulation provides robustness by bounding the influence of outlier residuals.

Theoretical analysis guarantees that #MTE possesses smooth decision boundaries, converges to hard rule ensembles in the limit, and can approximate continuous target functions on bounded domains. The Student-$t$ template ensures bounded score derivatives, offering a verifiable robustness advantage over Gaussian templates.

In synthetic experiments, #SMTE and #GMTE successfully isolate hidden mathematical logic and interactions. Ablation studies confirm that the signed log-margin is the primary driver of predictive performance, while the mask penalty directly controls structural sparsity. Across 28 tabular benchmarks, #SMTE achieves predictive performance competitive with strong tabular baselines like #EBM and #CatBoost while providing direct, rule-level interpretability. These results indicate that differentiable margin templates represent a robust and effective framework for interpretable tabular learning.

= Ethics and Broader Impact

Deploying interpretable models in high-stakes fields can improve transparency and algorithmic accountability. However, the ability to extract explicit rules does not guarantee systemic fairness or causal accuracy. Extracted rules can reflect historical biases present in the training data. We recommend thorough data auditing, bias evaluations, and domain-expert review before deploying these models in production environments.

= CRediT authorship contribution statement

= Data Statement

All datasets used in this work are publicly available or synthetically generated.

= Declaration of conflicts of interest

The author declares that no conflict of interest exists.

= Acknowledgment

This research did not receive external funding.

= Appendix A. Reproducibility Details

== Synthetic Datasets

The mathematical structures and active feature configurations for the synthetic tasks are detailed in @tab-synthetic-datasets.

#wide-table(
  compact-table[
    #table(
      columns: (1fr, .75fr, 1fr, 2.1fr),
      inset: (x: 2.5pt, y: 4.0pt),
      align: horizon,
      table.header(
        [*Dataset*], [*Task*], [*Active features*],
        [*Ground-truth structure*],
      ),
      [Hierarchical],
      [Regression],
      [$x_0, x_1, x_2, x_5, x_8$],
      [$3 * 1_(x_0 > 0.5, x_1 < 0) + 2 * 1_(x_2 > 1) x_5 + 1.5 x_8 + 0.8 x_0 x_1$],

      [Friedman1],
      [Regression],
      [$x_0, x_1, x_2, x_3, x_4$],
      [$10 sin(pi x_0 x_1) + 20(x_2 - 0.5)^2 + 10 x_3 + 5 x_4$],

      [XorChain],
      [Classification],
      [$x_0, x_1, x_2, x_3$],
      [Class 1: $x_0$ XOR $x_1$; class 2: $x_2 > 0.5, x_3 < -0.5$],

      [ThresholdCls],
      [Classification],
      [$x_0, x_4, x_7$],
      [$2 * 1_(x_0 > 0.3) - 1.5 * 1_(x_4 > 0) + x_7$],

      [MultiThresh],
      [Classification],
      [$x_0, x_1, x_2, x_6$],
      [Class threshold conjunctions],
    )
  ],
  [Synthetic dataset structures.]
) <tab-synthetic-datasets>

== Hyperparameters

The fixed hyperparameters used for the proposed models in both the synthetic and extended tabular benchmarks are listed in @tab-hyperparams. 

#wide-table(
  compact-table[
    #table(
      columns: (1.8fr, 1fr, 1fr),
      inset: 2.5pt,
      align: horizon,
      table.header(
        [*Setting*], [*Synthetic*], [*Extended benchmark*],
      ),
      [Rules ($R$)], [32], [64],
      [Epochs], [150], [200],
      [Patience], [20], [40],
      [Batch size], [256], [512],
      [Learning rate], [3e-3], [3e-3],
      [AdamW weight decay], [1e-4], [1e-4],
      [Mask penalty ($lambda_m$)], [1e-4], [1e-4],
      [Initial literal inverse temperature ($eta_kappa$)], [$log 6.0$], [$log 6.0$],
      [Allowed $kappa$ range], [[0.5, 50.0]], [[0.5, 50.0]],
      [Rule-activation sharpness ($beta$)], [6.0], [6.0],
      [Seeds], [1, 12, 123, 1234, 12345], [1, 12, 123, 1234, 12345],
    )
  ],
  [Optimization hyperparameters for the proposed #MTE architectures. The literal inverse temperature $kappa$ is learned from the listed initialization using a bounded exponential parameterization; $beta$ is fixed.]
) <tab-hyperparams>

== Ablation Protocol

The ablation study evaluates 15 configurations across the 5 synthetic datasets and 5 seeds, yielding 375 independent runs. All runs converged successfully.

The configurations include: (1) Reference #SMTE; (2) No signed evidence: fixing $s_(r j) = 1$; (3) No log-margin: using raw continuous literal values $ell_(r j)$ instead of log-margins $z_(r j)$; (4) No mask penalty: setting $lambda_m = 0$; (5) Fixed thresholds: keeping empirical quantile thresholds frozen during training; (6) Fixed scales: setting $tau_(r j) = 1$; and (7) Gaussian template: using Gaussian energy formulations. Throughout these ablations, the literal inverse temperature $kappa$ is learned and the rule-activation sharpness $beta$ remains fixed unless otherwise stated.

We evaluate sensitivity by varying $R in {16, 32, 64, 128}$ and $lambda_m in {0, 10^(-5), 10^(-4), 10^(-3)}$. The full protocol is defined in @tab-ablation-protocol.

#wide-table(
  compact-table[
    #table(
      columns: (1.4fr, 2.2fr),
      inset: 2.5pt,
      align: horizon,
      table.header([*Ablation*], [*Controlled change*]),

      [No signed evidence],
      [Fix $s_(r j) = 1$ across all rule-feature parameters.],

      [No log-margin],
      [Replace the signed log-margin $z_(r j)$ with the raw continuous literal $ell_(r j)$.],

      [No mask penalty],
      [Set the mask penalty parameter $lambda_m = 0$.],

      [Fixed thresholds],
      [Freeze thresholds at their initial empirical quantile values.],

      [Fixed scales],
      [Fix template scales to $tau_(r j) = 1$.],

      [Gaussian template],
      [Use Gaussian template energy functions instead of Student-$t$ formulations.],

      [Rule count],
      [Vary the number of rules across $R in {16, 32, 64, 128}$.],

      [Mask penalty],
      [Vary the sparsity penalty parameter across $lambda_m in {0, 10^(-5), 10^(-4), 10^(-3)}$.],
    )
  ],
  [Ablation configurations and sensitivity parameters.]
) <tab-ablation-protocol>