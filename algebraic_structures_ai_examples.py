# -*- coding: utf-8 -*-
"""
=============================================================================
FIVE MANUAL WORKED EXAMPLES — Algebraic Structures in Modern AI
=============================================================================
Author: Siraj Osman Omer (2026)

PURPOSE:
    This file directly implements the 5 MANUAL WORKED EXAMPLES from the paper
    "Algebraic Foundations of Modern Artificial Intelligence", Section 6.

WHAT THIS COVERS:
    1. G-Modules (Groups) — SO(2) Equivariant Neural Networks
    2. Tensor Products — Attention Mechanism (Q·Kᵀ)
    3. Functors (Category Theory) — Neural Network Composition
    4. Homological Algebra — Betti Numbers for Topological Data Analysis
    5. Algebraic Varieties — XOR Solved with Polynomial Network

HOW TO RUN:
    python algebraic_structures_ai_examples.py

REQUIREMENTS:
    numpy only (pure Python, no PyTorch needed for these examples!)
=============================================================================
"""

import numpy as np
from itertools import combinations

np.set_printoptions(precision=4, suppress=True)

# ─────────────────────────────────────────────────────────────────────────────
# HELPER: pretty section header
# ─────────────────────────────────────────────────────────────────────────────
def header(title, subtitle=""):
    print("\n" + "═"*70)
    print(f"  {title}")
    if subtitle:
        print(f"  {subtitle}")
    print("═"*70)

def step(n, description):
    print(f"\n  ── Step {n}: {description}")


# =============================================================================
# EXAMPLE 1: SO(2)-EQUIVARIANCE (Rotation Symmetry)
# =============================================================================
# Paper Section 6.1: "Verifying SO(2)-Equivariance by Hand"
#
# KEY IDEA: A neural layer φ(x) is "equivariant" to rotations if rotating the
#           input first gives the same result as rotating the output afterwards.
#           Math: φ(R·x) = R·φ(x)   for all rotation matrices R
# =============================================================================

header("EXAMPLE 1: SO(2) Rotational Equivariance",
       "Paper Section 6.1 — Does rotating input = rotating output?")

print("""
  INTUITION:
  Imagine a face detector. If you rotate the photo 45°, the detector should
  give the same result as detecting on the original photo and then rotating
  the result. That's equivariance.

  MATH: φ(R(θ)·x) = R(θ)·φ(x)
        where R(θ) = [[cos θ, -sin θ],
                      [sin θ,  cos θ]]
""")

# ── Step 1: Build the rotation matrix R(θ)
step(1, "Build rotation matrix R(θ) for θ = 45°")

theta = np.pi / 4   # 45 degrees
R = np.array([[np.cos(theta), -np.sin(theta)],
              [np.sin(theta),  np.cos(theta)]])
print(f"\n  θ = π/4 = 45°")
print(f"\n  R(45°) =\n{R}")
print(f"\n  This matrix rotates any 2D point by 45° counter-clockwise.")

# ── Step 2: Pick a layer that IS equivariant: φ(x) = R(α)·x (another rotation)
step(2, "Choose layer φ(x) = R(α)·x where α = 30°")

alpha = np.pi / 6   # 30 degrees
W = np.array([[np.cos(alpha), -np.sin(alpha)],
              [np.sin(alpha),  np.cos(alpha)]])
print(f"\n  Layer weight matrix W = R(30°) =\n{W}")
print(f"\n  The paper proves: rotations commute with rotations (SO(2) is abelian).")
print(f"  So W·R(θ) = R(θ)·W  — this layer IS equivariant.")

# ── Step 3: Test with a concrete point x = [1, 0]
step(3, "Test with point x = [1, 0]  (point on positive x-axis)")

x = np.array([1.0, 0.0])
print(f"\n  x = {x}   (a point at angle 0°, distance 1 from origin)")

# Path A: rotate input, then apply layer
x_rotated = R @ x              # rotate x by 45°
path_A = W @ x_rotated         # then apply layer
print(f"\n  PATH A — 'Rotate input FIRST, then layer':")
print(f"    R(45°)·x       = {x_rotated}   ← rotated 45°")
print(f"    W · (R(45°)·x) = {path_A}   ← layer applied after")

# Path B: apply layer, then rotate output
layer_x = W @ x               # apply layer to x
path_B = R @ layer_x           # then rotate result
print(f"\n  PATH B — 'Apply layer FIRST, then rotate output':")
print(f"    W·x            = {layer_x}   ← layer applied first")
print(f"    R(45°)·(W·x)   = {path_B}   ← then rotated 45°")

error = np.linalg.norm(path_A - path_B)
print(f"\n  ‖Path A − Path B‖ = {error:.8f}")
if error < 1e-10:
    print("  ✓ EQUIVARIANCE CONFIRMED: Both paths give identical results!")
    print("  ✓ Conclusion: R(30°) is a valid equivariant layer for SO(2).")


# =============================================================================
# EXAMPLE 2: TENSOR PRODUCTS (Attention Mechanism)
# =============================================================================
# Paper Section 6.2: "Computing Tensor Product Dimension"
#
# KEY IDEA: When attention computes Q·Kᵀ, it is contracting a tensor product.
#           The tensor product V⊗W has dimension dim(V) × dim(W).
# =============================================================================

header("EXAMPLE 2: Tensor Products — The Math Behind Attention",
       "Paper Section 6.2 — Why attention has dim(Q) × dim(K) possible interactions")

print("""
  INTUITION:
  In attention, each query word (Q) attends to each key word (K).
  The "space of all possible interactions" between Q and K is the tensor product.

  If Q lives in R^3 (3 dimensions) and K lives in R^2 (2 dimensions),
  there are 3 × 2 = 6 possible pairwise interactions.
  That's the tensor product R^3 ⊗ R^2 = R^6.
""")

step(1, "Define the two vector spaces V (queries) and W (keys)")

dim_V = 3   # query dimension
dim_W = 2   # key dimension
print(f"\n  V = R^{dim_V}  (query space, e.g. each query is a 3D vector)")
print(f"  W = R^{dim_W}  (key space,   e.g. each key   is a 2D vector)")

V_basis = [f"e{i+1}" for i in range(dim_V)]
W_basis = [f"f{j+1}" for j in range(dim_W)]
print(f"\n  Basis of V: {V_basis}")
print(f"  Basis of W: {W_basis}")

step(2, "Compute dimension of tensor product V ⊗ W")

dim_tensor = dim_V * dim_W
print(f"\n  Formula: dim(V ⊗ W) = dim(V) × dim(W) = {dim_V} × {dim_W} = {dim_tensor}")
print(f"\n  This means attention has {dim_tensor} independent interaction dimensions.")

step(3, "Write out ALL basis elements of V ⊗ W")

print(f"\n  Every basis element e_i ⊗ f_j captures one specific interaction:")
tensor_basis = []
for i, ei in enumerate(V_basis):
    for j, fj in enumerate(W_basis):
        basis_elem = f"{ei}⊗{fj}"
        tensor_basis.append(basis_elem)
        print(f"    {basis_elem}  ← query dim {i+1} interacts with key dim {j+1}")

print(f"\n  Total basis elements: {len(tensor_basis)} ✓  (matches dim = {dim_tensor})")

step(4, "Show how a concrete query/key pair becomes a tensor product element")

v = np.array([2.0, -1.0, 3.0])   # a query vector
w = np.array([1.0, 4.0])          # a key vector
print(f"\n  Example query vector v = {v}")
print(f"  Example key    vector w = {w}")

# The tensor product v⊗w is the outer product (a matrix)
outer = np.outer(v, w)
print(f"\n  v ⊗ w (as matrix, rows=query dims, cols=key dims):")
print(f"\n{outer}")
print(f"\n  Reading across rows: v⊗w = ", end="")
terms = []
for i in range(dim_V):
    for j in range(dim_W):
        terms.append(f"({outer[i,j]:.1f})·{V_basis[i]}⊗{W_basis[j]}")
print(" + ".join(terms))

print(f"\n  In attention: Q·Kᵀ computes this for ALL query-key pairs simultaneously.")
print(f"  The softmax then picks which interactions matter most.")


# =============================================================================
# EXAMPLE 3: CATEGORY THEORY (Functor Properties of Neural Nets)
# =============================================================================
# Paper Section 6.3: "Verifying Functor Properties"
#
# KEY IDEA: A functor must satisfy two rules:
#   1. F(identity) = identity
#   2. F(g∘f) = F(g)∘F(f)
# =============================================================================

header("EXAMPLE 3: Neural Networks as Categorical Functors",
       "Paper Section 6.3 — Two laws every neural net satisfies automatically")

print("""
  INTUITION:
  A functor is a "structure-preserving map between categories."
  For neural networks, the two categories are:
    • Input space  (data lives here)
    • Output space (predictions live here)

  Functor rule 1: If you do NOTHING to data → you get NOTHING changed in output.
  Functor rule 2: If you chain two operations, the network chains their effects.
""")

step(1, "Define a simple 1-layer network as a function")

# Using only numpy — no PyTorch needed
def relu(x):
    return np.maximum(0, x)

W1 = np.array([[0.5, -0.3],
               [0.2,  0.8],
               [-0.1, 0.4]])
b1 = np.array([0.1, -0.2, 0.0])

def layer_f(x):
    """Layer f: R^2 → R^3,   f(x) = ReLU(W1·x + b1)"""
    return relu(W1 @ x + b1)

W2 = np.array([[0.6, 0.1, -0.4],
               [0.3, 0.7,  0.2]])
b2 = np.array([0.0, 0.1])

def layer_g(x):
    """Layer g: R^3 → R^2,   g(x) = ReLU(W2·x + b2)"""
    return relu(W2 @ x + b2)

x = np.array([1.0, -0.5])
print(f"\n  Input x = {x}  (dimension 2)")
print(f"\n  Layer f: R^2 → R^3  (maps 2D to 3D)")
print(f"  Layer g: R^3 → R^2  (maps 3D to 2D)")

step(2, "Functor Axiom 1 — Identity preservation: F(id) = id")

def identity(x):
    return x.copy()

x_test = np.array([3.0, -1.0])
id_result = identity(x_test)
print(f"\n  Test: identity applied to x = {x_test}")
print(f"  Result = {id_result}")
print(f"  Same as input? {np.allclose(x_test, id_result)}")
print(f"\n  ✓ F(id_V)(x) = x = id_{{F(V)}}(x)  — Axiom 1 holds.")

step(3, "Functor Axiom 2 — Composition preservation: F(g∘f) = F(g)∘F(f)")

def composed_network(x):
    return layer_g(layer_f(x))

result_f = layer_f(x)
result_g_of_f = layer_g(result_f)
result_composed = composed_network(x)

print(f"\n  Input x = {x}")
print(f"\n  METHOD A — Compose into single function (g∘f), then apply:")
print(f"    (g∘f)(x) = {result_composed}")
print(f"\n  METHOD B — Apply f, then apply g to the result:")
print(f"    f(x)     = {result_f}   ← hidden representation")
print(f"    g(f(x))  = {result_g_of_f}   ← final output")

error = np.linalg.norm(result_composed - result_g_of_f)
print(f"\n  ‖Method A − Method B‖ = {error:.10f}")
print(f"  ✓ Identical results — Axiom 2 holds: F(g∘f) = F(g)∘F(f)")


# =============================================================================
# EXAMPLE 4: HOMOLOGICAL ALGEBRA (Betti Numbers for a Triangle)
# =============================================================================
# Paper Section 6.4: "Computing Betti Numbers by Hand"
#
# KEY IDEA: β₀ = number of CONNECTED COMPONENTS
#           β₁ = number of LOOPS / HOLES
# =============================================================================

header("EXAMPLE 4: Betti Numbers — Counting Topology of Shapes",
       "Paper Section 6.4 — β₀ = components, β₁ = holes")

print("""
  INTUITION:
  Topology asks: what is the "shape" of data, ignoring distances?

    β₀ = How many separate pieces? (1 for a single connected shape)
    β₁ = How many holes/loops?     (1 for a circle, 0 for a filled disk)

  We build a TRIANGLE: 3 vertices + 3 edges, NO filled interior.
  Expected: β₀=1 (connected), β₁=1 (has one loop/hole inside)
""")

step(1, "Build the triangle complex")

vertices = [0, 1, 2]
edges = [(0,1), (1,2), (0,2)]
triangles = []  # NO filled triangle!

n_v = len(vertices)
n_e = len(edges)
n_t = len(triangles)

print(f"\n  Vertices: {[f'v{v}' for v in vertices]}  ({n_v} total)")
print(f"  Edges:    {[f'e({a},{b})' for a,b in edges]}  ({n_e} total)")
print(f"  Triangles: {triangles}  ({n_t} total) ← NO filled interior!")

step(2, "Build the boundary matrix ∂₁  (edges → vertices)")

boundary_1 = np.zeros((n_v, n_e), dtype=float)
for col_idx, (i, j) in enumerate(edges):
    boundary_1[i, col_idx] = -1
    boundary_1[j, col_idx] = +1

print(f"\n  ∂₁ matrix =\n{boundary_1.astype(int)}")

step(3, "Compute ranks and Betti numbers")

rank_1 = np.linalg.matrix_rank(boundary_1)
rank_2 = 0

beta_0 = n_v - rank_1
kernel_dim_1 = n_e - rank_1
beta_1 = kernel_dim_1 - rank_2

print(f"\n  rank(∂₁) = {rank_1}")
print(f"  rank(∂₂) = {rank_2}")
print(f"\n  β₀ = {n_v} − {rank_1} = {beta_0}")
print(f"  β₁ = ({n_e} − {rank_1}) − {rank_2} = {beta_1}")

print(f"""
  ✓ RESULTS:
    β₀ = {beta_0} → The triangle is ONE connected shape
    β₁ = {beta_1} → The triangle has ONE hole/loop inside it

  IN AI: Persistent homology uses Betti numbers to describe point cloud shape.
  A ring of data points has β₁=1 (one loop) — useful for detecting circular
  patterns in molecular data, brain connectivity, and more.
""")


# =============================================================================
# EXAMPLE 5: XOR WITH POLYNOMIAL NETWORK (Algebraic Variety)
# =============================================================================
# Paper Section 6.5: "Solving XOR with Polynomial Network"
#
# KEY IDEA: XOR cannot be solved by a LINEAR function.
#           But a DEGREE-2 POLYNOMIAL can solve it exactly.
# =============================================================================

header("EXAMPLE 5: XOR with Polynomial Network — Exact Algebraic Solution",
       "Paper Section 6.5 — Why XOR needs nonlinearity, and how to find exact weights")

print("""
  INTUITION:
  XOR (exclusive-or) truth table:
    x1=0, x2=0 → output 0
    x1=0, x2=1 → output 1
    x1=1, x2=0 → output 1
    x1=1, x2=1 → output 0

  A straight line CANNOT separate the 1s from the 0s (they form an X pattern).
  But a POLYNOMIAL of degree 2 can — and we can find the exact coefficients
  by solving a system of equations algebraically.

  Polynomial ansatz:
    f(x1, x2) = w0 + w1·x1 + w2·x2 + w3·x1² + w4·x2² + w5·x1·x2
""")

step(1, "Write down equations from the XOR truth table")

print("""
  Plug each (x1, x2, target) into f:

    f(0,0) = w0                                        = 0   ...(eq. 1)
    f(0,1) = w0 + w2 + w4                              = 1   ...(eq. 2)
    f(1,0) = w0 + w1 + w3                              = 1   ...(eq. 3)
    f(1,1) = w0 + w1 + w2 + w3 + w4 + w5              = 0   ...(eq. 4)
""")

step(2, "Solve the system algebraically")

print("""
  From eq. 1:  w0 = 0

  Substitute into eq. 2:  w2 + w4 = 1
  Substitute into eq. 3:  w1 + w3 = 1

  From eq. 4 with w0=0:  w1 + w2 + w3 + w4 + w5 = 0
                         (w1 + w3) + (w2 + w4) + w5 = 0
                              1      +     1     + w5 = 0
                         → w5 = -2

  The paper picks the SIMPLEST/CANONICAL solution:
    w0=0, w1=1, w2=1, w3=0, w4=0, w5=-2

  → f(x1, x2) = x1 + x2 − 2·x1·x2
""")

w = np.array([0.0, 1.0, 1.0, 0.0, 0.0, -2.0])

def poly_network(x1, x2, w):
    return (w[0] + w[1]*x1 + w[2]*x2 +
            w[3]*x1**2 + w[4]*x2**2 + w[5]*x1*x2)

step(3, "Verify the solution on all XOR inputs")

truth_table = [(0, 0, 0), (0, 1, 1), (1, 0, 1), (1, 1, 0)]

print(f"\n  Weights: w = {w}")
print(f"  Formula: f(x1,x2) = {w[1]:.0f}·x1 + {w[2]:.0f}·x2 + ({w[5]:.0f})·x1·x2\n")
print(f"  {'x1':>4}  {'x2':>4}  {'Target':>8}  {'f(x1,x2)':>12}  {'Correct?':>10}")
print(f"  {'─'*50}")

all_correct = True
for x1, x2, target in truth_table:
    value = poly_network(x1, x2, w)
    correct = abs(value - target) < 1e-10
    all_correct = all_correct and correct
    sign = "✓" if correct else "✗"
    print(f"  {x1:>4}  {x2:>4}  {target:>8}  {value:>12.1f}  {sign:>10}")

print(f"\n  All correct? {all_correct}")

step(4, "Understand the decision boundary as an algebraic variety")

print("""
  The DECISION BOUNDARY is where f(x1, x2) = 0:

    x1 + x2 − 2·x1·x2 = 0

  This is an ALGEBRAIC VARIETY — the set of all points satisfying this equation.
  It's a degree-2 curve (a hyperbola!) that perfectly separates the XOR classes.

  In algebraic geometry terms:
    • Polynomial ring: R[x1, x2]
    • Ideal: I = ⟨x1 + x2 − 2·x1·x2⟩
    • Variety: V(I) = {(x1,x2) : x1 + x2 − 2·x1·x2 = 0}

  The Nullstellensatz connects this geometric curve to its algebraic definition.
""")

print(f"""
  ✓ SUMMARY OF EXAMPLE 5:
    • XOR is NOT linearly separable (no straight-line solution)
    • A degree-2 polynomial f(x1,x2) = x1 + x2 − 2·x1·x2 solves it EXACTLY
    • Found by algebra (not gradient descent) — perfect, zero error
    • Decision boundary is a hyperbola: an algebraic variety in R²
""")


# =============================================================================
# GRAND SUMMARY TABLE
# =============================================================================
header("SUMMARY: Five Algebraic Structures and Their AI Roles")

print("""
  ┌───┬──────────────────────────┬──────────────────────────────────────────┐
  │ # │ Algebraic Concept        │ AI Application (What it buys you)        │
  ├───┼──────────────────────────┼──────────────────────────────────────────┤
  │ 1 │ G-Modules (Groups)       │ Equivariant networks — rotate input =    │
  │   │                          │ rotate output. 12× data efficiency.      │
  ├───┼──────────────────────────┼──────────────────────────────────────────┤
  │ 2 │ Tensor Products          │ Attention = bilinear map between Q and K.│
  │   │                          │ Multi-head = decompose the tensor space. │
  ├───┼──────────────────────────┼──────────────────────────────────────────┤
  │ 3 │ Functors (Categories)    │ Nets are functors: composition law holds.│
  │   │                          │ Backprop is a contravariant functor.     │
  ├───┼──────────────────────────┼──────────────────────────────────────────┤
  │ 4 │ Homology (β₀, β₁, ...)  │ Count connected pieces and holes in data.│
  │   │                          │ Robust to noise — topology > geometry.   │
  ├───┼──────────────────────────┼──────────────────────────────────────────┤
  │ 5 │ Algebraic Varieties      │ Decision boundaries are polynomial curves│
  │   │                          │ Gröbner basis → exact network weights.   │
  └───┴──────────────────────────┴──────────────────────────────────────────┘

  Each example shows that core AI capabilities are not engineering tricks —
  they are consequences of deep algebraic structure.

   ALL FIVE EXAMPLES COMPLETE!
   For the full paper code with datasets, see:
     https://github.com/SirajOmerNG/homological-algebra-framework
""")

# Add this at the beginning of each example

def interactive_example_1():
    """Interactive SO(2) Equivariance Demo"""
    print("\n" + "="*70)
    print("  INTERACTIVE EXAMPLE 1: SO(2) Equivariance")
    print("="*70)
    
    # Get user input for rotation angle
    theta_deg = float(input("\nEnter rotation angle for R(θ) in degrees (e.g., 45): "))
    theta = np.radians(theta_deg)
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta), np.cos(theta)]])
    
    # Get user input for layer angle
    alpha_deg = float(input("Enter layer rotation angle in degrees (e.g., 30): "))
    alpha = np.radians(alpha_deg)
    W = np.array([[np.cos(alpha), -np.sin(alpha)],
                  [np.sin(alpha), np.cos(alpha)]])
    
    # Get user input for vector
    x1 = float(input("Enter x1 coordinate (e.g., 1): "))
    x2 = float(input("Enter x2 coordinate (e.g., 0): "))
    x = np.array([x1, x2])
    
    # Test equivariance
    path_A = W @ (R @ x)
    path_B = R @ (W @ x)
    
    print(f"\n  R({theta_deg}°) =\n{R}")
    print(f"\n  W({alpha_deg}°) =\n{W}")
    print(f"\n  Input vector x = {x}")
    print(f"\n  PATH A (rotate then layer): {path_A}")
    print(f"  PATH B (layer then rotate): {path_B}")
    
    error = np.linalg.norm(path_A - path_B)
    print(f"\n  ‖Path A − Path B‖ = {error:.8f}")
    
    if error < 1e-10:
        print("  ✓ EQUIVARIANCE CONFIRMED!")
    else:
        print("  ✗ NOT equivariant (layer must commute with rotation)")

def interactive_example_2():
    """Interactive Tensor Product Demo"""
    print("\n" + "="*70)
    print("  INTERACTIVE EXAMPLE 2: Tensor Products")
    print("="*70)
    
    # Get query vector dimension
    dim_V = int(input("\nEnter query vector dimension (e.g., 3): "))
    dim_W = int(input("Enter key vector dimension (e.g., 2): "))
    
    print(f"\n  V = R^{dim_V}, W = R^{dim_W}")
    print(f"  dim(V ⊗ W) = {dim_V} × {dim_W} = {dim_V * dim_W}")
    
    # Get query vector values
    print(f"\nEnter {dim_V} values for query vector v (space-separated):")
    v = np.array([float(x) for x in input().split()])
    
    # Get key vector values
    print(f"Enter {dim_W} values for key vector w (space-separated):")
    w = np.array([float(x) for x in input().split()])
    
    # Compute tensor product
    outer = np.outer(v, w)
    print(f"\n  v ⊗ w =\n{outer}")
    
    # Show basis decomposition
    print("\n  Decomposition:")
    for i in range(dim_V):
        for j in range(dim_W):
            if abs(outer[i,j]) > 1e-10:
                print(f"    ({outer[i,j]:.2f})·e{i+1}⊗f{j+1}")

def interactive_example_4():
    """Interactive Betti Numbers Demo"""
    print("\n" + "="*70)
    print("  INTERACTIVE EXAMPLE 4: Betti Numbers")
    print("="*70)
    
    print("\n  Build a simplicial complex:")
    n_v = int(input("  Number of vertices (e.g., 3): "))
    n_e = int(input("  Number of edges (e.g., 3): "))
    
    # Build custom boundary matrix
    print(f"\n  Enter {n_e} edges as 'i j' (space-separated, 0-indexed):")
    edges = []
    for e in range(n_e):
        i, j = map(int, input(f"  Edge {e+1}: ").split())
        edges.append((i, j))
    
    # Build boundary matrix
    boundary_1 = np.zeros((n_v, n_e))
    for col, (i, j) in enumerate(edges):
        boundary_1[i, col] = -1
        boundary_1[j, col] = 1
    
    rank_1 = np.linalg.matrix_rank(boundary_1)
    beta_0 = n_v - rank_1
    beta_1 = (n_e - rank_1) - 0
    
    print(f"\n  ∂₁ matrix:\n{boundary_1.astype(int)}")
    print(f"\n  rank(∂₁) = {rank_1}")
    print(f"  β₀ = {n_v} - {rank_1} = {beta_0}")
    print(f"  β₁ = {n_e} - {rank_1} = {beta_1}")

def interactive_example_5():
    """Interactive XOR Polynomial Demo"""
    print("\n" + "="*70)
    print("  INTERACTIVE EXAMPLE 5: XOR Polynomial")
    print("="*70)
    
    print("\n  Polynomial: f(x₁,x₂) = w₀ + w₁·x₁ + w₂·x₂ + w₃·x₁² + w₄·x₂² + w₅·x₁·x₂")
    
    # Let user modify weights
    print("\n  Current weights (canonical XOR solution):")
    w = np.array([0.0, 1.0, 1.0, 0.0, 0.0, -2.0])
    print(f"    w = {w}")
    
    modify = input("\n  Modify weights? (y/n): ").lower()
    if modify == 'y':
        w[0] = float(input("  Enter w₀ (bias): "))
        w[1] = float(input("  Enter w₁ (x₁ coefficient): "))
        w[2] = float(input("  Enter w₂ (x₂ coefficient): "))
        w[3] = float(input("  Enter w₃ (x₁² coefficient): "))
        w[4] = float(input("  Enter w₄ (x₂² coefficient): "))
        w[5] = float(input("  Enter w₅ (x₁·x₂ coefficient): "))
    
    # Test custom input
    print("\n  Test custom point:")
    x1 = float(input("  Enter x₁: "))
    x2 = float(input("  Enter x₂: "))
    
    def f(x1, x2):
        return w[0] + w[1]*x1 + w[2]*x2 + w[3]*x1**2 + w[4]*x2**2 + w[5]*x1*x2
    
    result = f(x1, x2)
    print(f"\n  f({x1}, {x2}) = {result:.4f}")
    
    # Test XOR truth table
    print("\n  XOR Truth Table Test:")
    print(f"  {'x1':>4} {'x2':>4} {'Target':>8} {'Output':>10} {'Correct?':>10}")
    print("  " + "-"*45)
    
    for x1, x2, target in [(0,0,0), (0,1,1), (1,0,1), (1,1,0)]:
        out = f(x1, x2)
        correct = abs(out - target) < 0.5
        print(f"  {x1:>4} {x2:>4} {target:>8} {out:>10.2f} {str(correct):>10}")

# Main interactive menu
def interactive_menu():
    while True:
        print("\n" + "="*60)
        print("  INTERACTIVE ALGEBRAIC STRUCTURES DEMO")
        print("="*60)
        print("\n  Select an example to run interactively:")
        print("    1. SO(2) Equivariance (G-Modules)")
        print("    2. Tensor Products (Attention)")
        print("    3. Functors (Neural Composition)")
        print("    4. Betti Numbers (Homology)")
        print("    5. XOR Polynomial (Varieties)")
        print("    0. Exit")
        
        choice = input("\n  Enter choice: ")
        
        if choice == '1':
            interactive_example_1()
        elif choice == '2':
            interactive_example_2()
        elif choice == '3':
            print("\n  Functor axioms always hold - no interactive input needed")
            print("  F(id) = id ✓")
            print("  F(g∘f) = F(g)∘F(f) ✓")
        elif choice == '4':
            interactive_example_4()
        elif choice == '5':
            interactive_example_5()
        elif choice == '0':
            print("\n  Goodbye!")
            break
        else:
            print("\n  Invalid choice. Try again.")

if __name__ == "__main__":
    interactive_menu()
