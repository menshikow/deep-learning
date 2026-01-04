# papers-list

### I. Chronological Timeline

**Foundations (1943–1997)**

* **1943** | **A Logical Calculus of the Ideas Immanent in Nervous Activity** — *McCulloch & Pitts*
* *The Concept:* The first formal mathematical model of a neural network.


* **1950** | **Computing Machinery and Intelligence** — *Alan Turing*
* *The Concept:* The Turing Test and the question "Can machines think?"


* **1958** | **The Perceptron: A Probabilistic Model for Information Storage** — *Rosenblatt*
* *The Concept:* The first practical algorithm for pattern recognition (single-layer).


* **1986** | **Learning Representations by Back-Propagating Errors** — *Rumelhart, Hinton, Williams*
* *The Concept:* **Backpropagation**. The algorithm that allows multi-layer networks to learn.


* **1997** | **Long Short-Term Memory (LSTM)** — *Hochreiter & Schmidhuber*
* *The Concept:* Solved the vanishing gradient problem for RNNs; enabled memory in sequence data.



**The Deep Learning Explosion (2012–2016)**

* **2012** | **ImageNet Classification with Deep CNNs (AlexNet)** — *Krizhevsky et al.*
* *The Concept:* Proved CNNs could dominate computer vision with GPU training.


* **2013** | **Efficient Estimation of Word Representations (Word2Vec)** — *Mikolov et al.*
* *The Concept:* Vector embeddings ().


* **2013** | **Playing Atari with Deep Reinforcement Learning (DQN)** — *Mnih et al.*
* *The Concept:* The marriage of Deep Learning and Reinforcement Learning.


* **2014** | **Generative Adversarial Networks (GANs)** — *Goodfellow et al.*
* *The Concept:* Generator vs. Discriminator. The birth of modern generative AI.


* **2014** | **Adam: A Method for Stochastic Optimization** — *Kingma & Ba*
* *The Concept:* The default optimizer used in almost every modern neural network.


* **2015** | **Deep Residual Learning for Image Recognition (ResNet)** — *He et al.*
* *The Concept:* Skip connections allow networks to scale from 20 layers to 1000+.



**Transformers & The Age of Scale (2017–2020)**

* **2017** | **Attention Is All You Need** — *Vaswani et al.*
* *The Concept:* The **Transformer** architecture. Displaced RNNs/LSTMs entirely.


* **2017** | **Model-Agnostic Meta-Learning (MAML)** — *Finn et al.*
* *The Concept:* A framework for "learning to learn" new tasks quickly.


* **2018** | **BERT: Pre-training of Deep Bidirectional Transformers** — *Devlin et al.*
* *The Concept:* Self-supervised learning (masking) for deep language understanding.


* **2020** | **Language Models are Few-Shot Learners (GPT-3)** — *Brown et al.*
* *The Concept:* At massive scale, models exhibit emergent behavior (few-shot learning).


* **2020** | **Denoising Diffusion Probabilistic Models (DDPM)** — *Ho et al.*
* *The Concept:* The foundation behind DALL-E 2, Midjourney, and Stable Diffusion.



**Alignment, Agents & Structure (2021–Present)**

* **2021** | **Geometric Deep Learning** — *Bronstein et al.*
* *The Concept:* Unifying CNNs, GNNs, and Transformers through symmetry and invariance.


* **2022** | **Training Compute-Optimal Large Language Models (Chinchilla)** — *Hoffmann et al.*
* *The Concept:* Corrected scaling laws; data quality/quantity matters as much as model size.


* **2022** | **Training Language Models to Follow Instructions (InstructGPT)** — *Ouyang et al.*
* *The Concept:* **RLHF** (Reinforcement Learning from Human Feedback). Turned raw LLMs into Chatbots.



---

### II. Priority Order (Tiered Reading)

**Tier 1: The Axioms (Read First)**

> *If you only read 5 papers, read these to understand how everything works.*

1. **Backpropagation (1986)** – The mechanism of learning.
2. **Attention Is All You Need (2017)** – The architecture of modern AI.
3. **ResNet (2015)** – The structural breakthrough permitting depth.
4. **Adam (2014)** – The optimization standard.
5. **InstructGPT (2022)** – The alignment method that makes AI usable.

**Tier 2: The Paradigms (Conceptual Deepening)**

1. **DQN (2013)** – Foundations of Deep RL.
2. **Word2Vec (2013)** – Foundations of semantic space.
3. **Diffusion Models (2020)** – Foundations of image generation.
4. **Chinchilla (2022)** – Foundations of scaling laws/economics.
5. **AlexNet (2012)** – The historic turning point.

**Tier 3: Systems & Frontiers**

1. **MAML (2017)** – Meta-learning.
2. **Geometric DL (2021)** – Graph and structural theory.
3. **GANs (2014)** – Adversarial training dynamics.
4. **PPO / SAC** – Modern RL Policy Optimization.

---

### III. Suggested Study Path (Practical Order)

**Phase I: Mechanics**

* *Backpropagation*  *Adam Optimizer*  *Word2Vec*

**Phase II: Vision & Depth**

* *AlexNet*  *ResNet*  *GANs*

**Phase III: The Transformer Era**

* *Attention Is All You Need*  *BERT*  *GPT-3*

**Phase IV: Modern LLMs**

* *Chinchilla (Scaling)*  *InstructGPT (Alignment/RLHF)*

**Phase V: Advanced Horizons**

* *Diffusion Models*  *Geometric DL*  *Multi-Agent RL*

