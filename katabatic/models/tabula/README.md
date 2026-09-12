# TaBuLa

**TaBuLa** is a hybrid generative model for tabular data that combines **latent-variable modeling with diffusion-style refinement**. It is designed to capture complex feature dependencies but comes with **high computational and memory requirements**, particularly for medium-to-large datasets.

In this project, TaBuLa was explored as a **representative VAE–diffusion hybrid model** within the Katabatic generative benchmarking framework.

***

## Overview

TaBuLa aims to improve tabular data synthesis by:
- Learning a latent representation of tabular data
- Applying diffusion-inspired transformations to refine samples
- Preserving complex feature interactions beyond standard GAN-based approaches

While theoretically powerful, TaBuLa is **computationally intensive** and sensitive to implementation and hardware constraints.

***

## Implementation Journey and Challenges

### Initial Self-Implementation Attempt

A significant amount of time was spent attempting to **implement TaBuLa from scratch**, following the original paper and public descriptions.

During this phase:
- Core architectural components were implemented manually
- Multiple training configurations were tested
- The model consistently failed to converge or produced unstable outputs

After extensive debugging and validation attempts, it was concluded that the self-implemented version was **not behaving reliably** and could not be trusted for fair evaluation.

As a result, the self-implemented code was **intentionally removed** to avoid introducing incorrect or misleading results into the project.

---

### Pivot to Official Repository Implementation

To ensure correctness and alignment with the original method, the approach was revised to **utilize the official TaBuLa class from the authors’ repository**.

This decision was made to:
- Avoid deviating from the reference implementation
- Ensure algorithmic correctness
- Focus evaluation on model behavior rather than implementation errors

The reused components were integrated into the Katabatic pipeline where possible, while preserving attribution to the original authors.

---

### Hardware and Scalability Constraints

Despite using the official implementation, TaBuLa proved to be **computationally heavys** under the available hardware constraints.

Specifically:
- Training time increased sharply with dataset size
- Memory usage exceeded practical limits on several datasets
- Full multi-dataset evaluation was not feasible within project timelines

As a result:
- TaBuLa was **successfully executed only on the Car dataset**
- Experiments on larger datasets were intentionally discontinued

This limitation is explicitly acknowledged and factored into result interpretation.

***

## Experimental Scope

- **Datasets evaluated**: Car only
- **Evaluation method**: TSTR (Train on Synthetic, Test on Real)
- **Role in project**: Demonstrates trade-offs between model expressiveness and computational feasibility

TaBuLa results are included for **qualitative and methodological comparison**, not as a fully competitive baseline across all datasets.

***

## Deviations from the Original Paper

Relative to the original TaBuLa paper, this implementation differs in the following ways:

1. **Partial dataset coverage** due to hardware limitations
2. **No architectural modifications**, relying on the official implementation
3. **Controlled experimental scope**, limited to feasibility testing
4. **Pipeline-level integration only**, without extensive hyperparameter tuning

These deviations were necessary to maintain experimental integrity under real-world constraints.

***

## References

- Reference paper: https://arxiv.org/abs/1604.04960
- Reference repository: https://github.com/zhao-zilong/Tabula

***

## Source Code Reuse and Attribution

This implementation makes **direct use of the TaBuLa class and supporting modules from the official repository**.

All reused code:
- Remains attributed to the original authors
- Is used in accordance with the repository license
- Was not reimplemented unnecessarily to avoid introducing deviations

The author’s contribution focuses on **evaluation design, pipeline integration, and experimental analysis**.

***

## Generative AI Acknowledgement

**ChatGPT (OpenAI)** was used to assist with interpreting the TaBuLa paper and repository, and to support reasoning about implementation and experimental design.

All generated guidance was **manually verified** against the original sources. The final decisions regarding implementation scope, code reuse, and result inclusion were made independently by the author.
# TabuLa

**TabuLa** is a language-model-based approach for synthetic tabular data
generation. It represents tabular rows as text, trains a causal language model
on the resulting token sequences, and generates synthetic rows that are parsed
back into tabular data.

## Overview

The TabuLa workflow is:

1. Convert tabular rows into textual representations.
2. Randomise column order during training.
3. Tokenise the textual rows.
4. Train a causal language model.
5. Generate new textual rows.
6. Parse the generated text back into tabular data.

The Katabatic integration preserves the original TabuLa implementation as much
as possible while adapting it to Katabatic's model interface.

## Implementation

The implementation is integrated into:

```text
katabatic/models/tabula/
├── __init__.py
├── models.py
├── utils.py
└── README.md
