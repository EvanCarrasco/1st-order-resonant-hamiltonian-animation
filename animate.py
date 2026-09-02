import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

fig, ax = plt.subplots(1,1,figsize=(4, 4))

## based on code written by Mutian Wang
def plot_hamiltonian_Nesvorny16(
    delta,
    PSI_max,
    ax,
):
    x = np.linspace(-4 * PSI_max, 4 * PSI_max, 2000)
    y = np.linspace(-4 * PSI_max, 4 * PSI_max, 2000)
    X, Y = np.meshgrid(x, y)
    PSI = np.sqrt(X**2 + Y**2)
    psi = np.arctan2(Y, X)
    
    # Hamiltonian
    H = -((PSI - delta) ** 2) - np.sqrt(2 * PSI) * np.cos(psi)

    ax.set_title(r"$\delta=${0:.2f}".format(delta))
    ax.set_xlabel(r"$\Psi\cos(\psi)$")
    ax.set_ylabel(r"$\Psi\sin(\psi)$")

    # Solve for equilibrium roots: Psi^3 - 2*delta*Psi^2 + delta^2*Psi - 1/8 = 0
    poly_coeff = [1, -2 * delta, delta**2, -1 / 8]
    roots = np.roots(poly_coeff)
    
    # Safely filter real roots (ignoring small imaginary artifacts)
    real_psi_roots = roots[np.abs(roots.imag) < 1e-7].real
    real_psi_roots = np.sort(real_psi_roots[real_psi_roots > 0])

    if len(real_psi_roots) < 3:
        # Single equilibrium point case (below resonant bifurcation)
        psi_center = real_psi_roots[0]
        ax.plot(-psi_center, 0, "r+", ms=6, zorder=10)
        
        H_levels = np.linspace(np.min(H), np.max(H), 20)
        ax.contour(X, Y, H, levels=H_levels, colors="k", linewidths=0.5,linestyles="solid")

    else:
        # Three equilibrium points case (in resonance)
        # Psi_1 < delta (psi=0, stable)
        # Psi_2 < delta (psi=0, unstable)
        # Psi_3 > delta (psi=pi, stable)
        psi_1, psi_2, psi_3 = real_psi_roots
        
        stable1_x = -psi_3  # at psi = pi
        stable2_x = psi_1   # at psi = 0
        unstable_x = psi_2  # at psi = 0

        # Plot equilibrium points
        ax.plot(stable1_x, 0, "r+", ms=6, zorder=10)
        ax.plot(stable2_x, 0, "r+", ms=6, zorder=10)
        ax.plot(unstable_x, 0, "b+", ms=6, zorder=10)

        # Calculate energy levels at equilibrium points
        H_stable1 = -((psi_3 - delta) ** 2) + np.sqrt(2 * psi_3)
        H_stable2 = -((psi_1 - delta) ** 2) - np.sqrt(2 * psi_1)
        H_unstable = -((psi_2 - delta) ** 2) - np.sqrt(2 * psi_2)

        # Plot separatrix directly using contour at energy level H_unstable
        ax.contour(X, Y, H, levels=[H_unstable], colors="r", linewidths=1.2, zorder=5, linestyles="solid")

        # Define background contour levels avoiding duplicates
        H_levels = np.hstack(
            [
                np.linspace(np.min(H), H_stable2, 5, endpoint=False),
                np.linspace(H_unstable, H_stable2, 5)[1:],
                np.linspace(H_unstable, H_stable1, 6, endpoint=False),
            ]
        )
        H_levels = np.unique(np.sort(H_levels))
        
        ax.contour(X, Y, H, levels=H_levels, colors="k", linewidths=0.5, linestyles="solid")

def update(frame, PSI_max=2, ax=ax):
    delta = -1.5 + (frame / 240) * 5.5
    ax.clear()
    plot_hamiltonian_Nesvorny16(delta, PSI_max, ax=ax)


anim = FuncAnimation(fig, update, frames=241, interval=50)
anim.save("resonant_hamiltonian.gif", writer="pillow", fps=20, dpi=300)

plt.tight_layout()
plt.show()