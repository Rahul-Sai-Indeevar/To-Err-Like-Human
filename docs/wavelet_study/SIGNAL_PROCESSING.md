# From sampled images to a two-level Haar filter bank

Our signal is a sampled RGB photograph. We ask whether a two-channel Haar filter bank behaves as predicted by discrete-time signal processing, and whether giving its complementary outputs to separate classifier branches helps visual emotion recognition. These questions use different measurements. Reconstruction, subband energy, alias cancellation and shift equivariance test the transform. Held-out Emotion Confusion Confidence (ECC) tests recognition.

The assigned paper, *To Err Like Human*, gives an image credit $1/W_{ij}$ when true emotion $i$ is predicted as $j$, where $W_{ij}$ is the paper's emotion distance. ECC averages this credit over all images. An exact prediction gets full credit and an emotionally close mistake gets partial credit. Our filters act on pixels upstream of the classifier. The paper's metric evaluates the resulting predictions. [Assigned NeurIPS paper](https://proceedings.neurips.cc/paper_files/paper/2024/file/f2bb120e9a2cb9c2a50921b7f865c421-Paper-Conference.pdf).

## 1. CTFT, sampling, DTFT and 2D FFT

Reason about one color channel because the same spatial transform acts independently on red, green and blue. Let $x_c(u,v)$ be a continuous image, with vertical and horizontal coordinates $u,v$. Its continuous spatial Fourier transform is

$$
X_c(\Omega_r,\Omega_c)=
\int_{\mathbb R}\int_{\mathbb R}x_c(u,v)
e^{-j(\Omega_r u+\Omega_c v)}\,du\,dv.
$$

With sample spacings $T_r,T_c$, the image lattice is $x[r,c]=x_c(rT_r,cT_c)$. Under ideal point sampling, the 2D DTFT is a sum of shifted CTFT copies

$$
X(\omega_r,\omega_c)=
\frac1{T_rT_c}\sum_{p,q\in\mathbb Z}
X_c\!\left(\frac{\omega_r-2\pi p}{T_r},
           \frac{\omega_c-2\pi q}{T_c}\right).
$$

The DTFT is $2\pi$-periodic along each axis, and $\omega=2\pi f$ converts cycles per pixel to radians per pixel. If the pre-sampling spectrum lies inside each axis's Nyquist region, the copies remain separate. Overlap sums different continuous frequencies at the same discrete frequency. The available photographs already live on the pixel lattice. Our alias experiment studies a **new decimation inside the wavelet bank** on that lattice.

For a finite $H\times W$ crop, the 2D DFT samples the crop's DTFT at $(2\pi k/H,2\pi l/W)$:

$$
X[k,l]=\sum_{r=0}^{H-1}\sum_{c=0}^{W-1}x[r,c]\,
e^{-j2\pi(kr/H+lc/W)}.
$$

The FFT computes this DFT efficiently. Our earlier spatial experiment applied radial low-pass masks in this domain and inverse-transformed the results. On full EmoSet, mean ECC of three frozen cross-entropy models went from 80.225% on clean images to 45.736% at 0.05 cycles/pixel and 71.868% at 0.20 cycles/pixel. Removing spatial frequencies affected predictions made by those fixed models. We then asked whether a model could learn useful features from separately available coarse and detail signals. [Spatial FFT experiment](../full_spatial_frequency/REPORT.md).

## 2. DTFT of the Haar analysis filters

Haar maps adjacent even and odd samples into a scaled sum and a signed difference:

$$
\begin{bmatrix}a[m]\\d[m]\end{bmatrix}
=\frac1{\sqrt2}
\begin{bmatrix}1&1\\1&-1\end{bmatrix}
\begin{bmatrix}x[2m]\\x[2m+1]\end{bmatrix}.
$$

Equivalently, form the local sum and difference at every index and retain every second result. With the forward-neighbor convention used here,

$$
H_L(\omega)=\frac{1+e^{j\omega}}{\sqrt2},
\qquad H_H(\omega)=\frac{1-e^{j\omega}}{\sqrt2}.
$$

The positive exponent arises because $x[n+1]$ is a one-sample advance. A causal delay convention changes phase and can change the high branch's sign. The squared magnitudes stay the same:

$$
|H_L(\omega)|^2=1+\cos\omega=2\cos^2(\omega/2),\qquad
|H_H(\omega)|^2=1-\cos\omega=2\sin^2(\omega/2).
$$

At DC the adjacent samples agree, so the difference vanishes. At Nyquist they alternate, so the sum vanishes. At $\omega=\pi/2$ both filters have unit squared gain before decimation. The decimator halves the sample count, so the corresponding **energy fractions** for one complex sinusoid are $|H_L|^2/2$ and $|H_H|^2/2$. They sum to one. A maximum squared filter gain of two makes sense because the output band has half as many samples.

## 3. Downsampling, aliasing and perfect reconstruction

Sampling at even indices maps $\omega$ to $2\omega$ modulo $2\pi$. The tones $e^{j\omega n}$ and $e^{j(\omega+\pi)n}$ both become $e^{j2\omega m}$, since $e^{j2\pi m}=1$. Haar can still distinguish them through the pair of complex gains supplied by the two filters. The DTFT statement of the folding is

$$
Y_i(\Omega)=\frac12\left[
H_i(\Omega/2)X(\Omega/2)
+ H_i(\Omega/2+\pi)X(\Omega/2+\pi)
\right],\qquad i\in\{L,H\}.
$$

For synthesis, insert a zero between successive samples of each band, then apply matched synthesis filters. The low coefficient contributes with equal signs at even and odd output positions. The high coefficient contributes with opposite signs. Thus $G_L=(1+e^{-j\omega})/\sqrt2$ and $G_H=(1-e^{-j\omega})/\sqrt2$. The upsamplers give $Y_i(2\omega)$, and the combined output is

$$
\tilde X(\omega)=
\underbrace{\tfrac12\left[G_L H_L+G_H H_H\right]X(\omega)}_{\text{desired}}
+\underbrace{\tfrac12\left[G_L H_L(\omega+\pi)+G_H H_H(\omega+\pi)\right]X(\omega+\pi)}_{\text{alias}}.
$$

Each $G_i$ and each unshifted $H_i$ is evaluated at $\omega$. The desired gain is $\tfrac12(|H_L|^2+|H_H|^2)=1$. The two alias contributions are

$$
G_LH_L(\omega+\pi)=\frac{e^{-j\omega}-e^{j\omega}}2,\qquad
G_HH_H(\omega+\pi)=\frac{e^{j\omega}-e^{-j\omega}}2.
$$

They cancel exactly. This works along both image axes. The inverse in the code uses equivalent even/odd interleaving formulas. A single reconstructed branch generally retains an alias term. Recombining the full bank recovers the input. The [experiment appendix](wavelet_sp_experiments.md) records the measured collision and cancellation using DFT-bin tones. [Multiresolution filter-bank reference](https://www.di.ens.fr/~mallat/papiers/MallatTheory89.pdf).

## 4. Separable 2D analysis and two levels

Filtering along rows and columns, with decimation by two on each axis, gives four subbands. We use the first letter for the row axis and the second for the column axis. In one $2\times2$ block, $a,b$ are the top row and $c,d$ the bottom row:

$$
LL=(a+b+c+d)/2,\quad LH=(a-b+c-d)/2,\quad
HL=(a+b-c-d)/2,\quad HH=(a-b-c+d)/2.
$$

Each axis contributes a factor $1/\sqrt2$. Vertical stripes vary with column position and can excite $LH$. A checkerboard varies along both axes and can excite $HH$. The implementation slices the four array phases directly. Its signed-sum matrix has orthonormal rows, so the transpose inverts it and preserves squared norm. For $\begin{bmatrix}6&2\\4&0\end{bmatrix}$, the coefficients are $(6,4,2,0)$, and both energies are 56. Synthesis with LL gives a constant block of 3. The other bands synthesize $\begin{bmatrix}3&-1\\1&-3\end{bmatrix}$. [Analysis and synthesis code](../../src/ecc/wavelets.py).

For a bin-centered complex tone $e^{j(\omega_r r+\omega_c c)}$, the first-level band energy fraction is

$$
F_{b_r b_c}=\tfrac14|H_{b_r}(\omega_r)|^2|H_{b_c}(\omega_c)|^2.
$$

The product comes from separability and $1/4$ from decimating in both axes. The four fractions sum to one. The numerical experiment passes cosine and sine through the real code as two channels. Adding their measured energies gives the complex-tone energy and avoids phase-dependent cross terms at frequencies that fold together.

At level two, run the bank again on $LL_1$, which now has frequency $2\omega$ on each axis. The level-two approximation energy fraction for the same input tone is

$$
F_{LL_2}=\cos^2(\omega_r/2)\cos^2(\omega_r)
          \cos^2(\omega_c/2)\cos^2(\omega_c).
$$

Three first-level detail bands, three second-level detail bands and $LL_2$ contain $3HW/4+3HW/16+HW/16=HW$ coefficients per channel. The 160×160 crop gives bands of 80×80 at level one and 40×40 at level two. [Training protocol](STUDY_PLAN.md).

## 5. What the model actually receives

Keeping only $LL_2$ in synthesis returns one 4×4 block mean repeated at every pixel of that block. In vector form the operator on one block is

$$
P=\frac1{16}\mathbf1\mathbf1^{\mathsf T},\qquad
P^{\mathsf T}=P,\qquad P^2=P.
$$

The approximation is $x_L=Px$. The signed detail is $x_D=(I-P)x$, which contains all six detail bands and has zero mean within each block. Since $P(I-P)=0$,

$$
x=x_L+x_D,\qquad
\langle x_L,x_D\rangle=0,\qquad
\|x\|^2=\|x_L\|^2+\|x_D\|^2.
$$

The implementation first applies the pretrained model's affine normalization $z=(x-\mu)/\sigma$ per channel. A constant belongs to the approximation subspace, so $Pz=(Px-\mu)/\sigma$ and $(I-P)z=(x-Px)/\sigma$. The code retains the sign and scale of the residual. Rectification, clipping or subtracting an additional channel mean from the detail would change the decomposition. [Component implementation](../../src/ecc/wavelets.py).

For zero-mean white noise $n$ with covariance $\sigma^2I$, the expected block energy after projection is

$$
\mathbb E\|Pn\|^2=\operatorname{tr}(P\sigma^2 I)=\sigma^2.
$$

The original block has expected energy $16\sigma^2$. Therefore $1/16=6.25\%$ goes into the approximation and $15/16=93.75\%$ into the detail. The noise samples in the reconstructed approximation are correlated within a block because they share its average. With colored noise of covariance $\Sigma$, the expected coarse energy is $\operatorname{tr}(P\Sigma)$. This prediction is about the input perturbation. Recognition also depends on the useful image structure removed by averaging.

## 6. Removing the block origin gives a new FIR filter

For a circular horizontal shift $T_s$, quantify whether the anchored transform commutes with the shift:

$$
\eta_s(x)=\frac{\|PT_sx-T_sPx\|_2}{\|x\|_2}.
$$

A one-pixel shift moves some samples across a block boundary. The two orders can give different outputs. A shift of four pixels preserves 4×4 groupings on the divisible circular grid, so $PT_4=T_4P$. This is **shift equivariance**: the output shifts with the input. The anchored block projection is periodically shift variant. Its tone energy fractions and the FIR analysis gains describe different operations.

Average the projections over **all 16 alignments**, undoing each shift before averaging:

$$
P_{\mathrm{avg}}=
\frac1{16}\sum_{a,b=0}^{3}
T_{(-a,-b)}P T_{(a,b)}.
$$

On the periodic grid, a shift simply permutes the alignments. The resulting operator commutes with every circular shift. Its exact convolution kernel follows from counting how often two positions share a block. Positions at separation $k$ share a block in $4-|k|$ of four one-dimensional alignments for $|k|<4$. Each contributing mean weights the neighbor by $1/4$. Therefore the one-dimensional kernel and its DTFT are

$$
w[k]=\frac{(4-|k|)_+}{16},
\qquad w[-3:3]=\frac1{16}[1,\ 2,\ 3,\ 4,\ 3,\ 2,\ 1],
$$

$$
W(\omega)=\left|\frac14\sum_{m=0}^{3}e^{-j\omega m}\right|^2
=\left[\frac{\sin(2\omega)}{4\sin(\omega/2)}\right]^2,\qquad W(0)=1.
$$

The triangle is a four-tap box convolved with its reversal. Its 2D kernel is $w[r]w[c]$ and the response is $W(\omega_r)W(\omega_c)$. The wider support removes grid alignment dependence and increases smoothing. For white noise, the output variance per pixel is the input variance times the kernel's squared norm. On a periodic image large enough that all seven taps have distinct locations per axis, the retained energy fraction is

$$
\left[\frac{1+4+9+16+9+4+1}{16^2}\right]^2
=\left(\frac{11}{64}\right)^2=\frac{121}{4096}\approx2.9541\%.
$$

Phase averaging retains exact additive reconstruction as $P_{\mathrm{avg}}x+(I-P_{\mathrm{avg}})x=x$. Its frequency response has gains strictly between zero and one, whose values change upon squaring. Consequently $P_{\mathrm{avg}}^2\ne P_{\mathrm{avg}}$ and the two components generally have a nonzero inner product. The energy partition of Section 5 belongs to the original orthogonal bank. The [experiment appendix](wavelet_sp_experiments.md) reports numerical checks on the averaged operator. The existing classifiers were trained with the anchored representation.

## 7. What the measured evidence establishes

| Prediction | Original measurement | Interpretation |
|---|---|---|
| The full bank reconstructs and preserves energy. | For 256 training images per dataset in float64, the reconstruction and subband sum had zero measured residual and maximum relative Parseval error $6.43\times10^{-16}$. | The implemented sums and inverse agree with the algebra on these tensors. |
| The anchored approximation receives 6.25% of white-noise energy. | A synthetic realization gave 6.2179% approximation and 93.7821% detail. | The finite-sample split agrees with the approximation subspace's rank. |
| The grid changes after a one-pixel circular shift and returns to alignment after four. | Mean one-pixel commutators on FI, the EmoSet subset and ArtPhoto were 0.19658, 0.20914 and 0.16967. Four-pixel residuals were zero. | Grid dependence is measurable before the classifier sees the image. |
| The coarse experiment can show a smaller ECC drop under added noise. | At Gaussian $\sigma=0.03$, coarse-only FI lost 0.535 ECC points versus 1.790 for RGB. On the EmoSet subset it lost 0.578 versus 3.206. Absolute noisy ECC still favored RGB on both splits. | The coarse model degraded less from a lower clean baseline. Both the change and the final score matter. |

[Numerical diagnostics and frozen-model protocol](DIAGNOSTICS.md).

The training comparison kept cross-entropy and training budgets fixed. Six input arrangements and eight fresh paired seeds on three datasets gave 144 model fits. Both coarse + signed detail and the primary RGB + RGB control used 22,361,224 parameters, with matched starting weights where applicable, example orders and augmentations within each seed. Validation ECC selected checkpoints. The test ECC comparison was:

| Prepared test split | Coarse + detail ECC | RGB + RGB ECC | Paired difference, percentage points | Paired seed 95% interval |
|---|---:|---:|---:|---:|
| FI | 63.309% | 66.332% | −3.023 | [−4.124, −1.922] |
| EmoSet subset | 68.953% | 73.104% | −4.151 | [−4.846, −3.455] |
| ArtPhoto | 50.657% | 51.625% | −0.968 | [−2.713, +0.778] |

Fusion lost all eight FI and EmoSet pairs. ArtPhoto's interval spans zero. The intervals estimate paired seed variation on each prepared split. Perfect reconstruction says that a known linear inverse can recover the original from both branches. The classifiers must separately learn from differently distributed inputs, so that identity makes no prediction about the direction of the ECC effect. [Completed wavelet findings](FINDINGS.md).

## Code and verification

- The [filter bank and models](../../src/ecc/wavelets.py) implement analysis and synthesis in **haar_analysis2d** and **haar_synthesis2d**. **wavelet_components** supplies the two model inputs.
- The [study configuration](../../configs/wavelet_study.yaml), [study runner](../../scripts/run_wavelet_study.py) and [transform tests](../../tests/test_wavelets.py) document the original experiment. The [project guide](../../README.md) maps the runs and verification records.
- The [experiment appendix](wavelet_sp_experiments.md) documents the new train-free diagnostics, including tone response, alias cancellation and the phase-averaged FIR.
