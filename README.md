# deep-learning

*This roadmap is inspired by the structure and approach found in [adam-maj/deep-learning](https://github.com/adam-maj/deep-learning).*

## Table of Contents

1. [Chronological Timeline](https://www.google.com/search?q=%23i-chronological-timeline)
2. [Priority Order](https://www.google.com/search?q=%23ii-priority-order)

---

## I. Chronological Timeline

### Foundations (1943–1997)

*The Era of Logic and Early Networks*

* **1943 | A Logical Calculus of the Ideas Immanent in Nervous Activity** — *McCulloch & Pitts*
* **Concept:** The first formal mathematical model of a neural network.


* **1950 | Computing Machinery and Intelligence** — *Alan Turing*
* **Concept:** The Turing Test and the question "Can machines think?"


* **1958 | The Perceptron** — *Rosenblatt*
* **Concept:** The first practical algorithm for pattern recognition (single-layer).


* **1986 | Learning Representations by Back-Propagating Errors** — *Rumelhart, Hinton, Williams*
* **Concept:** Backpropagation. The algorithm that allows multi-layer networks to learn.



* **1997 | Long Short-Term Memory (LSTM)** — *Hochreiter & Schmidhuber*
* **Concept:** Solved the vanishing gradient problem for RNNs; enabled memory in sequence data.



### The Deep Learning Explosion (2012–2016)

*The Era of Vision, Optimization, and Vectors*

* **2012 | AlexNet** — *Krizhevsky et al.*
* **Concept:** Proved CNNs could dominate computer vision with GPU training.


* **2013 | Word2Vec** — *Mikolov et al.*
* **Concept:** Vector embeddings ().


* **2013 | Playing Atari with Deep RL (DQN)** — *Mnih et al.*
* **Concept:** The marriage of Deep Learning and Reinforcement Learning.


* **2014 | Generative Adversarial Networks (GANs)** — *Goodfellow et al.*
* **Concept:** Generator vs. Discriminator. The birth of modern generative AI.



* **2014 | Adam Optimizer** — *Kingma & Ba*
* **Concept:** The default optimizer used in almost every modern neural network.


* **2015 | ResNet** — *He et al.*
* **Concept:** Skip connections allow networks to scale from 20 layers to 1000+.



### Transformers & The Age of Scale (2017–2020)

*The Era of Attention and Generative Pre-training*

* **2017 | Attention Is All You Need** — *Vaswani et al.*
* **Concept:** The Transformer architecture. Displaced RNNs/LSTMs entirely.


* **2017 | Model-Agnostic Meta-Learning (MAML)** — *Finn et al.*
* **Concept:** A framework for "learning to learn" new tasks quickly.


* **2018 | BERT** — *Devlin et al.*
* **Concept:** Self-supervised learning (masking) for deep language understanding.


* **2020 | GPT-3** — *Brown et al.*
* **Concept:** At massive scale, models exhibit emergent behavior (few-shot learning).


* **2020 | Denoising Diffusion Probabilistic Models (DDPM)** — *Ho et al.*
* **Concept:** The foundation behind DALL-E 2, Midjourney, and Stable Diffusion.



### Alignment, Agents & Structure (2021–Present)

*The Era of Instructions, Reasoning, and Safety*

* **2021 | Geometric Deep Learning** — *Bronstein et al.*
* **Concept:** Unifying CNNs, GNNs, and Transformers through symmetry and invariance.


* **2022 | Chinchilla** — *Hoffmann et al.*
* **Concept:** Corrected scaling laws; data quality/quantity matters as much as model size.


* **2022 | InstructGPT** — *Ouyang et al.*
* **Concept:** RLHF (Reinforcement Learning from Human Feedback). Turned raw LLMs into Chatbots.



---

## II. Priority Order

If time is limited, follow this priority list to grasp the core mechanisms of the field.

| Tier | Focus | Key Papers |
| --- | --- | --- |
| **Tier 1** | **The Axioms** | **Backprop** (Mechanism), **Attention** (Architecture), **ResNet** (Depth), **Adam** (Optimization), **InstructGPT** (Alignment) |
| **Tier 2** | **The Paradigms** | **DQN** (RL), **Word2Vec** (Semantics), **Diffusion** (GenAI), **Chinchilla** (Scaling), **AlexNet** (History) |
| **Tier 3** | **Frontiers** | **MAML** (Meta), **Geometric DL** (Structure), **GANs** (Adversarial), **PPO/SAC** (Agents) |
