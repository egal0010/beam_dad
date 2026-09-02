# Deep Adaptive Design for Beam Alignment

## Bayesian Optimal Experimental Design

**Bayesian Optimal Experimental Design (BOED)** provides a principled framework for deciding **which experiment should be performed next** in order to learn as much as possible about an unknown parameter $\theta$.

Given a possible experimental design $\xi$, its usefulness can be quantified through the **Expected Information Gain (EIG)**:

$$
\mathrm{EIG}(\xi)
=
\mathbb{E}_{p(\theta)p(y\mid\theta,\xi)}
\left[
\log
\frac{p(y\mid\theta,\xi)}
{p(y\mid\xi)}
\right].
$$

where

$$
p(y\mid\xi)
=
\int
p(y\mid\theta,\xi)p(\theta)\,d\theta.
$$

The optimal experiment is therefore

$$
\xi^\star
=
\arg\max_{\xi}
\mathrm{EIG}(\xi).
$$

Intuitively, we want to choose the experiment whose observation is expected to reduce our uncertainty about $\theta$ the most.

---

## The Computational Problem

Although the formulation is simple, evaluating the EIG is generally difficult.

The expectation requires averaging over both the unknown parameter $\theta$ and the possible observation $y$, while the term

$$
p(y\mid\xi)
=
\int
p(y\mid\theta,\xi)p(\theta)\,d\theta
$$

requires an additional integration over the parameter space.

A natural solution is therefore to use a **nested Monte Carlo estimator**. However, doing so for every candidate experiment can quickly become computationally expensive.

This becomes particularly problematic for **real-time sequential systems**, where a new experiment must be selected after every observation.

---

## Application: Beam Alignment and Localization

This project investigates BOED in the context of **beamforming-based device localization**.

The objective is to estimate the **Angle of Arrival (AoA)** of a transmitter using as few measurements as possible.

At each time step, the system:

1. selects a beam $\mathbf{b}_t$,
2. performs a measurement $\mathbf{r}_t$,
3. updates its knowledge about the transmitter direction,
4. selects the next beam based on all previous measurements.

The history available at time $t$ can therefore be written as

$$
h_t
=
\left\{
(\mathbf{b}_1,\mathbf{r}_1),
\dots,
(\mathbf{b}_t,\mathbf{r}_t)
\right\}.
$$

In principle, the next beam could be selected by solving a new Bayesian experimental design problem after every measurement. However, repeatedly estimating the EIG online is too computationally expensive for many real-time applications.

---

## Deep Adaptive Design

**Deep Adaptive Design (DAD)** addresses this problem by moving most of the computational cost **offline**.

Instead of explicitly solving the Bayesian design problem at every time step, DAD trains a neural network to represent an experimental design **policy**:

$$
\mathbf{b}_{t+1}
=
\pi_{\phi}(h_t).
$$

where

* $h_t$ is the complete measurement history,
* $\pi_{\phi}$ is a neural network parameterized by $\phi$,
* $\mathbf{b}_{t+1}$ is the next beam to transmit.

The network is trained so that the sequence of experiments it generates maximizes the expected information gained about the unknown parameter.

Once training is complete, experiment selection no longer requires an expensive online EIG optimization: the next beam is obtained through a simple forward pass through the policy network.

This makes DAD particularly attractive for **adaptive beam alignment and localization in real-time wireless systems**.
