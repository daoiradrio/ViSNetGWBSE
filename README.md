# Transfer learning of GW-BSE excitation energies and qsGW quasiparticle energies

Repository to the paper "Transfer learning of GW Bethe-Salpeter Equation excitation energies" (LINK). We show how DFT and TDDFT data can be leveraged to 1) reduce test errors of GNN-based predictions of qsGW quasiparticle energies and GW-BSE excitation energies and 2) reduce the required amount of costly qsGW and GW-BSE needed for doing so.

The filenames of model checkpoints of pretrained models in the `checkpoints/` directory follow the pattern:

pre-`prop`-<code>n<sub>pre</sub></code>-M-<code>n<sub>feat</sub></code>-<code>r<sub>cut</sub></code>-<code>n<sub>bas</sub></code>-<code>n<sub>mp</sub></code>-<code>l<sub>max</sub></code>

For instance, `prehomo1M128502532` denotes a model pretrained on 1,000,000 (`1M`) HOMO energies (`prop`) with 128 feature channels (<code>n<sub>feat</sub></code>), using a cutoff distance of 5.0 Å (<code>r<sub>cut</sub></code> with the decimal point left out in the filenames), 25 radial basis functions (<code>n<sub>bas</sub></code>), 3 message-passing layers (<code>n<sub>mp</sub></code>) and <code>l<sub>max</sub></code> of 2. The filenames of pretrained and finetuned models follow the pattern:

`prop`-<code>n<sub>pre</sub></code>-M-<code>n<sub>feat</sub></code>-<code>r<sub>cut</sub></code>-<code>n<sub>bas</sub></code>-<code>n<sub>mp</sub></code>-<code>l<sub>max</sub></code>

For instance, `homo1M128502532` denotes a model pretrained and finetuned on HOMO energies. The filenames of the baseline models without pretraining, so 0 pretraining samples, and only finetuning follow the pattern:

`prop`-0M-<code>n<sub>feat</sub></code>-<code>r<sub>cut</sub></code>-<code>n<sub>bas</sub></code>-<code>n<sub>mp</sub></code>-<code>l<sub>max</sub></code>
