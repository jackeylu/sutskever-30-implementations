# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This repository contains **30 comprehensive Jupyter notebook implementations** of Ilya Sutskever's recommended deep learning papers. All implementations use **pure NumPy only** (no PyTorch/TensorFlow) with **synthetic/bootstrapped data** for immediate execution and educational clarity.

**Status**: All 30 papers implemented (100% complete)

**Implementation Philosophy**:
- **NumPy-only**: Every operation is explicit for maximum educational value
- **Synthetic data**: No external dataset downloads required; each notebook generates its own data
- **Self-contained**: Each notebook can be run independently
- **Visualizations**: Extensive plots and explanations for each concept

## Quick Start Commands

### Environment Setup
```bash
# Install dependencies (minimal requirements)
pip install numpy matplotlib scipy jupyter

# Start Jupyter to run any notebook
jupyter notebook
```

**Note**: There are no build scripts, test suites, or linting configurations. The repository consists solely of educational Jupyter notebooks that can be run individually.

### Running Notebooks
```bash
# Run a specific notebook
jupyter notebook 02_char_rnn_karpathy.ipynb

# Or use Jupyter Lab for better IDE experience
jupyter lab
```

### Common Development Tasks

**Testing a notebook implementation**:
```bash
# Open notebook in Jupyter and run all cells
# Use "Kernel -> Restart & Run All" to verify complete execution
```

**Checking for broken notebooks**:
```bash
# Use nbconvert to test if notebooks execute cleanly
jupyter nbconvert --to notebook --execute 02_char_rnn_karpathy.ipynb
```

## Repository Architecture

### Structure
```
sutskever-30-implementations/
├── README.md                    # Main documentation with paper list
├── IMPLEMENTATION_TRACKS.md     # Detailed implementation track for each paper
├── 01_*.ipynb through 30_*.ipynb  # 30 paper implementations
└── *.py, *.png, *.npz           # Supporting files for Paper 18 (Relational RNN)
```

### Notebook Naming Convention
Notebooks are numbered `01_` through `30_` corresponding to the Sutskever reading list order. Each is self-contained with:
- Paper title and authors in the first cell
- Pure NumPy implementation (no frameworks)
- Synthetic data generation
- Visualization sections
- "Key Takeaways" summary section

### Paper Groupings

**Foundational Concepts (Papers 1-5)**:
- Complexity dynamics, RNN basics, LSTM, Regularization, Pruning/MDL

**Architectures & Mechanisms (Papers 6-15)**:
- Pointer Networks, AlexNet, Seq2Seq for Sets, GPipe, ResNet, Dilated Convolutions, GNNs, Transformers, Bahdanau Attention, Pre-activation ResNet

**Advanced Topics (Papers 16-22)**:
- Relational Reasoning, VAE, Relational RNN, Coffee Automaton (Irreversibility), Neural Turing Machines, CTC, Scaling Laws

**Theory & Meta-Learning (Papers 23-30)**:
- MDL Principle, Machine Super Intelligence, Kolmogorov Complexity, CS231n, Multi-token Prediction, Dense Retrieval, RAG, Lost in the Middle

### Featured Comprehensive Notebooks

Some papers have **multi-section comprehensive implementations**:

1. **Paper 18 (Relational RNN)**: Includes Section 11 with manual backpropagation (~1100 lines of gradient code)
2. **Paper 19 (Coffee Automaton)**: 10 sections deep-diving into irreversibility, entropy, Landauer's principle (~2500 lines)
3. **Paper 24 (Machine Super Intelligence)**: 6 sections covering Universal AI, AIXI, Solomonoff induction (~2000 lines)
4. **Paper 26 (CS231n)**: Complete vision pipeline from kNN to CNNs in 10 sections (~2400 lines)

These are the **longest and most complex** notebooks in the repository.

### Code Patterns

**Typical notebook structure**:
```python
# 1. Imports (only numpy, matplotlib, scipy)
import numpy as np
import matplotlib.pyplot as plt

# 2. Synthetic data generation
data = generate_synthetic_data(...)

# 3. Model implementation (class-based)
class ModelName:
    def __init__(self, ...):
        # Initialize weights (typically * 0.01 for small values)
        self.W = np.random.randn(...) * 0.01

    def forward(self, ...):
        # Explicit forward pass
        ...

    def backward(self, ...):
        # Manual gradient computation
        ...

# 4. Training loop
def train(model, data, ...):
    # Often uses Adagrad or simple SGD
    # Gradient clipping (typically to [-5, 5])
    ...

# 5. Visualization
plt.figure(figsize=(...))
plt.plot(...)
plt.show()

# 6. Key Takeaways (markdown cell)
```

**Common conventions**:
- **Random seed**: `np.random.seed(42)` for reproducibility
- **Weight initialization**: `np.random.randn(...) * 0.01` (small random values)
- **Gradient clipping**: `np.clip(dparam, -5, 5, out=dparam)` to prevent exploding gradients
- **Loss tracking**: Smoothed loss with `smooth_loss = smooth_loss * 0.999 + loss * 0.001`
- **Optimizer**: Adagrad is commonly used (memory-based adaptive learning rates)

## Key Implementation Insights

### What Makes These Implementations Unique

1. **No Framework Abstractions**: Every forward/backward pass is explicit. You can see exactly how gradients flow through each operation.

2. **Synthetic Data Strategy**: Each notebook generates its own data. Examples:
   - Text data: Repeated phrases or patterns
   - Image data: Procedural patterns (shapes, textures)
   - Graph data: Synthetic molecular structures
   - This means **zero setup time** - notebooks just work

3. **Educational Clarity Over Performance**:
   - Code prioritizes readability
   - Extensive comments explain each step
   - Visualizations show intermediate states
   - "Key Takeaways" sections summarize core concepts

### Inter-Paper Connections

The implementation tracks reveal important conceptual links:
- **Papers 5, 23, 25**: Pruning → MDL → Kolmogorov Complexity (compression trilogy)
- **Papers 7, 10, 15, 26**: AlexNet → ResNet → Pre-activation → CS231n (vision evolution)
- **Papers 2, 3, 4, 18**: Char RNN → LSTM → Regularization → Relational RNN (sequence modeling)
- **Papers 14, 13, 6**: Bahdanau Attention → Transformers → Pointer Networks (attention evolution)
- **Papers 23, 24, 25**: MDL → Universal AI → Kolmogorov (intelligence theory)

### Implementation Difficulty Levels

**Beginner (afternoon projects)**:
- 02 (Char RNN), 04 (RNN Regularization), 05 (Pruning), 07 (AlexNet), 10 (ResNet), 17 (VAE)

**Intermediate (weekend projects)**:
- 03 (LSTM), 06 (Pointer Networks), 08 (Seq2Seq for Sets), 12 (GNNs), 14 (Bahdanau), 16 (Relation Networks), 18 (Relational RNN), 22 (Scaling Laws), 27-30 (modern techniques)

**Advanced (week-long deep dives)**:
- 09 (GPipe), 13 (Transformer), 20 (NTM), 29 (RAG)

**Comprehensive/Theoretical** (multi-section explorations):
- 01 (Complexity Dynamics), 19 (Coffee Automaton - 10 sections), 23 (MDL), 24 (Machine Super Intelligence - 6 sections), 25 (Kolmogorov), 26 (CS231n - 10 sections)

## Working with This Codebase

### Adding New Paper Implementations

When implementing a new paper:

1. **Follow the naming convention**: `XX_paper_title.ipynb` where XX is the paper number
2. **Use the standard structure**:
   - Paper title and authors in first markdown cell
   - NumPy-only implementation
   - Synthetic data generation
   - Visualizations
   - Key Takeaways summary
3. **Update IMPLEMENTATION_TRACKS.md** with the implementation details
4. **Update README.md** if needed (paper list, learning paths)
5. **Test the notebook** by running all cells from top to bottom

### Modifying Existing Notebooks

**Important**: These are **educational implementations**, not production code. When modifying:
- Preserve the **educational clarity** - don't "optimize" at the cost of readability
- Keep **synthetic data generation** - don't add external dataset dependencies
- Maintain **NumPy-only** approach - don't introduce PyTorch/TensorFlow
- Update **Key Takeaways** section if you add new insights

### Debugging Notebook Issues

If a notebook doesn't run correctly:
1. Check for missing imports (should only be `numpy`, `matplotlib`, `scipy`)
2. Verify random seed is set (`np.random.seed(42)`)
3. Look for shape mismatches in matrix operations
4. Check gradient clipping is applied
5. Verify data generation produces expected output shapes

### Understanding Manual Backpropagation

Several notebooks (especially Paper 18) include **manual backpropagation implementations**. These are intentionally verbose to show every gradient step. When reading:
- Focus on understanding the **chain rule** applications
- Note how gradients flow through time (BPTT) or layers
- Gradient clipping is essential for stability
- Adagrad or SGD is typically used for updates

## Supporting Files (Paper 18 Only)

**Note**: Paper 18 (Relational RNN) has additional Python files:
- `relational_rnn_cell.py`: Core RNN cell implementation
- `relational_memory.py`: Memory mechanism
- `reasoning_tasks.py`: Task definitions
- `training_utils.py`: Training utilities
- `*.py` test files: Validation scripts
- `*.npz`, `*.png`: Saved models and visualizations

These are **specific to Paper 18** and were part of the original implementation. Other notebooks are fully self-contained.

## Common Pitfalls

1. **Assuming framework usage**: This is pure NumPy - no `torch.nn`, no `tf.keras`
2. **Looking for test suites**: There are no automated tests - notebooks are verified by manual execution
3. **Expecting real datasets**: All data is synthetic - no downloads needed
4. **Wanting production performance**: These are educational implementations prioritizing clarity over speed
5. **Missing the educational intent**: Every explicit operation is intentional for learning

## Reference Materials

- **README.md**: Complete paper list, learning paths, quick reference
- **IMPLEMENTATION_TRACKS.md**: Detailed implementation track for each of the 30 papers
- Each notebook's "Key Takeaways" section: Core concept summaries

## Citation

If you reference these implementations:
```bibtex
@misc{sutskever30implementations,
  title={Sutskever 30: Complete Implementation Suite},
  author={Paul "The Pageman" Pajo},
  year={2025},
  note={Educational implementations of Ilya Sutskever's recommended reading list}
}
```

## Key Principle

**"If you really learn all of these, you'll know 90% of what matters today."** - Ilya Sutskever

This repository exists to make those 30 papers accessible through **hands-on implementation** rather than just reading. Every line of code is written to be **understood**, not just run.
