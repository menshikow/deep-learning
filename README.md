# deep-learning

*This roadmap is inspired by the structure and approach found in [adam-maj/deep-learning](https://github.com/adam-maj/deep-learning).*

## Table of Contents

1. [Chronological Timeline](https://www.google.com/search?q=%23i-chronological-timeline)
2. [Priority Order](https://www.google.com/search?q=%23ii-priority-order)

## **I. Foundations (1943–1997)**

*The Era of Logic and Early Neural Networks*

* **1943 | A Logical Calculus of the Ideas Immanent in Nervous Activity** — *McCulloch & Pitts*  
  First formal mathematical model of a neural network.  
  📄 [PDF](https://www.cs.cmu.edu/afs/cs.cmu.edu/project/ai-repository/ai/areas/reasoning/mcp43.pdf)

* **1950 | Computing Machinery and Intelligence** — *Alan Turing*  
  Introduced the Turing Test; posed the question "Can machines think?"  
  📄 [PDF](https://www.csee.umbc.edu/courses/471/papers/turing.pdf)

* **1958 | The Perceptron** — *Rosenblatt*  
  First practical algorithm for pattern recognition (single-layer network).

* **1986 | Learning Representations by Back-Propagating Errors** — *Rumelhart, Hinton & Williams*  
  Backpropagation algorithm enabling multi-layer neural networks.  
  📄 [PDF](http://www.cs.toronto.edu/~hinton/absps/backprop.pdf)

* **1997 | Long Short-Term Memory (LSTM)** — *Hochreiter & Schmidhuber*  
  Solved vanishing gradient problem in RNNs; introduced memory in sequence learning.  
  📄 [PDF](https://www.bioinf.jku.at/publications/older/2604.pdf)

---

## **II. The Deep Learning Explosion (2012–2016)**

*Vision, optimization, and deep models*

* **2012 | ImageNet Classification with Deep Convolutional Neural Networks (AlexNet)** — *Krizhevsky et al.*  
  CNNs trained on GPUs revolutionized computer vision.  
  📄 [PDF](https://papers.nips.cc/paper/2012/hash/c399862d3b9d6b76c8436e924a68c45b-Abstract.html)

* **2013 | Efficient Estimation of Word Representations in Vector Space (Word2Vec)** — *Mikolov et al.*  
  Learned continuous vector embeddings capturing semantic word relationships.  
  📄 [arXiv:1301.3781](https://arxiv.org/abs/1301.3781)

* **2013 | Playing Atari with Deep Reinforcement Learning (DQN)** — *Mnih et al.*  
  Combined deep learning and RL to play Atari games.  
  📄 [arXiv:1312.5602](https://arxiv.org/abs/1312.5602)

* **2014 | Generative Adversarial Networks (GANs)** — *Goodfellow et al.*  
  Generator vs. discriminator for realistic data generation.  
  📄 [arXiv:1406.2661](https://arxiv.org/abs/1406.2661)

* **2014 | Adam: A Method for Stochastic Optimization** — *Kingma & Ba*  
  Popular optimizer for neural networks.  
  📄 [arXiv:1412.6980](https://arxiv.org/abs/1412.6980)

* **2015 | Deep Residual Learning for Image Recognition (ResNet)** — *He et al.*  
  Skip connections allow very deep networks to train.  
  📄 [arXiv:1512.03385](https://arxiv.org/abs/1512.03385)

---

## **III. Transformers & The Age of Scale (2017–2020)**

*Attention, pre-training, and emergent behaviors*

* **2017 | Attention Is All You Need** — *Vaswani et al.*  
  Transformer architecture replaces RNNs for sequence modeling.  
  📄 [arXiv:1706.03762](https://arxiv.org/abs/1706.03762)

* **2017 | Model-Agnostic Meta-Learning (MAML)** — *Finn et al.*  
  Meta-learning framework for fast adaptation to new tasks.  
  📄 [arXiv:1703.03400](https://arxiv.org/abs/1703.03400)

* **2018 | BERT: Pre-training of Deep Bidirectional Transformers** — *Devlin et al.*  
  Self-supervised masked language modeling for NLP tasks.  
  📄 [arXiv:1810.04805](https://arxiv.org/abs/1810.04805)

* **2019 | AlphaZero: Mastering Chess and Shogi by Self-Play** — *Silver et al.*  
  Generalized AlphaGo self-play approach across multiple games.  
  📄 [arXiv:1712.01815](https://arxiv.org/abs/1712.01815)

* **2020 | Language Models are Few-Shot Learners (GPT-3)** — *Brown et al.*  
  Large-scale transformer exhibits few-shot capabilities.  
  📄 [arXiv:2005.14165](https://arxiv.org/abs/2005.14165)

* **2020 | Denoising Diffusion Probabilistic Models (DDPMs)** — *Ho et al.*  
  Foundation for modern generative image models (DALL-E 2, Stable Diffusion).  
  📄 [arXiv:2006.11239](https://arxiv.org/abs/2006.11239)

---

## **IV. AlphaGo / Game AI Milestones (2016–2020)**

*Bridging planning and learning*

* **2016 | Mastering the Game of Go with Deep Neural Networks and Tree Search (AlphaGo)** — *Silver et al.*  
  Neural nets + Monte Carlo Tree Search defeated human champion.  
  📄 [Nature](https://www.nature.com/articles/nature16961)

* **2020 | Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model (MuZero)** — *Schrittwieser et al.*  
  Learns environment model and plans without prior knowledge of rules.  
  📄 [arXiv:1911.08265](https://arxiv.org/abs/1911.08265)

---

## **V. Alignment, Scaling, Agents & Structure (2021–Present)**

*Emergence, structured representations, and human alignment*

* **2021 | Geometric Deep Learning** — *Bronstein et al.*  
  Unifies CNNs, GNNs, and transformers through geometric structures.

* **2022 | Chinchilla: Rethinking Model Scaling** — *Hoffmann et al.*  
  Scaling laws emphasize data quality and balance with model size.  
  📄 [arXiv:2203.15556](https://arxiv.org/abs/2203.15556)

* **2022 | Training Language Models to Follow Instructions with Human Feedback (InstructGPT)** — *Ouyang et al.*  
  RLHF aligns models with human preferences.  
  📄 [arXiv:2203.02155](https://arxiv.org/abs/2203.02155)

* **2020s | Scaling Laws for Compute & Performance** — *Kaplan et al.*  
  Empirical evidence: scaling compute and data drives performance.  
  📄 [arXiv:2001.08361](https://arxiv.org/abs/2001.08361)

* **2020s | Bitter Lesson (Sutton)** — *Richard Sutton*  
  Emphasizes general-purpose learning systems over human-engineered heuristics.  
  📄 [PDF](http://www.incompleteideas.net/IncIdeas/BitterLesson.html)

---

## 🚀 **Usage Tip for Repo**

- Copy this Markdown into `README.md` or `timeline.md`.  
- Each entry can be **linked to your own implementation** if you recreate it from scratch.  
- Optionally, add **tags** like `#vision`, `#RL`, `#LLM`, `#meta-learning`, `#generative`.