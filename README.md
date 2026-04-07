# Algebraic Structures in Modern AI - Five Worked Examples

[![GitHub stars](https://img.shields.io/github/stars/SirajOmerNG/algebraic-structures-ai-examples)](https://github.com/SirajOmerNG/algebraic-structures-ai-examples/stargazers)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

## 📖 Overview

This repository contains **five fully-worked examples** demonstrating how abstract algebra underpins modern artificial intelligence.

Based on the paper: *"Algebraic Foundations of Modern Artificial Intelligence"* (Siraj Osman Omer, 2026)

## Examples Included

| # | Algebraic Concept | AI Application | Key Result |
|---|-------------------|----------------|-------------|
| 1 | G-Modules (Groups) | SO(2) Equivariant Networks | φ(R·x) = R·φ(x) ✓ |
| 2 | Tensor Products | Attention Mechanism | dim(V⊗W) = 3×2 = 6 |
| 3 | Functors (Category Theory) | Neural Network Composition | F(g∘f) = F(g)∘F(f) ✓ |
| 4 | Homological Algebra | Betti Numbers for TDA | β₀=1, β₁=1 |
| 5 | Algebraic Varieties | XOR Polynomial Solution | f = x₁ + x₂ - 2x₁x₂ |

##  How to Run

```bash
git clone https://github.com/SirajOmerNG/algebraic-structures-ai-examples.git
cd algebraic-structures-ai-examples
pip install -r requirements.txt
python algebraic_structures_ai_examples.py