import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Initialize figure with 3D projection
fig = plt.figure(figsize=(9, 7))
ax = fig.add_subplot(111, projection='3d')

## Based on code written by Mutian Wang
def plot_hamiltonian_Nesvorny16_3D(delta, PSI_max, ax):
    x = np.linspace(-4 * PSI_max, 4 * PSI_max, 200)
    y = np.linspace(-4 * PSI_max, 4 * PSI_max, 200)
    X, Y = np.meshgrid(x, y)
    PSI = np.sqrt(X**2 + Y**2)
    psi = np.arctan2(Y, X)

    # Hamiltonian energy surface
    H = -((PSI - delta) ** 2) - np.sqrt(2 * PSI) * np.cos(psi)

    # Fixed bottom and top bounds
    z_top = -130.0
    z_bottom = 10.0
    ax.set_zlim(z_bottom, z_top)

    # 1. Plot 3D Energy Surface FIRST
    surf = ax.plot_surface(
        X, Y, H, cmap="viridis", alpha=0.75, rcount=100, ccount=100, antialiased=True
    )

    # 2. Project background contours onto the floor SECOND
    ax.contour(
        X,
        Y,
        H,
        levels=20,
        cmap="binary",
        linestyles="solid",
        offset=z_bottom,
        linewidths=0.5,
    )

    ax.set_title(r"Hamiltonian Topology Surface ($\delta=${0:.2f})".format(delta))
    ax.set_xlabel(r"$\Psi\cos(\psi)$")
    ax.set_ylabel(r"$\Psi\sin(\psi)$")
    ax.set_zlabel(r"$H(\Psi, \psi)$")

    # Solve for equilibrium points
    poly_coeff = [1, -2 * delta, delta**2, -1 / 8]
    roots = np.roots(poly_coeff)
    real_psi_roots = roots[np.abs(roots.imag) < 1e-7].real
    real_psi_roots = np.sort(real_psi_roots)

    # Lists to collect fixed points for scatter plotting at the end
    surface_points = []  # tuples of (x, y, z_surface, color)
    floor_points = []    # tuples of (x, y, color)

    if len(real_psi_roots) < 3:
        # Single stable center below bifurcation
        psi_center = real_psi_roots[0] * -1
        H_center = -((-psi_center - delta) ** 2) + np.sqrt(2 * abs(psi_center))
        
        surface_points.append((psi_center, 0, H_center, "red"))
        floor_points.append((psi_center, 0, "red"))

    else:
        # Three equilibrium points in resonance
        psi_1, psi_2, psi_3 = real_psi_roots
        stable1_x = -psi_3
        stable2_x = psi_1
        unstable_x = psi_2

        H_stable1 = -((-stable1_x - delta) ** 2) + np.sqrt(2 * abs(stable1_x))
        H_stable2 = -((stable2_x - delta) ** 2) - np.sqrt(2 * abs(stable2_x))
        H_unstable = -((unstable_x - delta) ** 2) - np.sqrt(2 * abs(unstable_x))

        surface_points.append((stable1_x, 0, H_stable1, "red"))
        surface_points.append((stable2_x, 0, H_stable2, "red"))
        surface_points.append((unstable_x, 0, H_unstable, "blue"))

        floor_points.append((stable1_x, 0, "red"))
        floor_points.append((stable2_x, 0, "red"))
        floor_points.append((unstable_x, 0, "blue"))

        # 3. Draw 3D Separatrix line directly on the surface
        ax.contour(
            X,
            Y,
            H,
            levels=[H_unstable - 2.0],
            colors="red",
            linewidths=1.2,
            zorder=5,
            linestyles="solid",
        )
        
        # 4. Draw projected separatrix level set on the floor
        ax.contour(
            X,
            Y,
            H,  # Offset to project onto the floor
            levels=[H_unstable],
            colors="red",
            linewidths=1.2,
            offset=z_bottom,
            zorder=5,
            linestyles="solid",
        )


    # 5. DRAW ALL SCATTER POINTS LAST WITH SMALL Z-OFFSETS TO PREVENT OCCLUSION
    z_surf_offset = 2.0    # Lifts 3D markers slightly above the surface mesh

    for x_p, y_p, z_p, col in surface_points:
        ax.scatter(
            [x_p],
            [y_p],
            [z_p - z_surf_offset],
            color=col,
            s=50,
            edgecolor="black",
            linewidth=0.5,
            zorder=20,
            depthshade=False,
        )

    for x_p, y_p, col in floor_points:
        ax.scatter(
            [x_p],
            [y_p],
            [z_bottom],
            color=col,
            marker="+",
            s=70,
            linewidths=2.0,
            zorder=20,
            depthshade=False,
        )

    # Fixed 3D camera viewing angle
    ax.view_init(elev=30, azim=-65)

    # Hide z-axis tick labels
    ax.set_zticklabels([])


def update(frame, PSI_max=2, ax=ax):
    delta = -1.5 + (frame / 240) * 5.5
    ax.clear()
    plot_hamiltonian_Nesvorny16_3D(delta, PSI_max, ax=ax)


anim = FuncAnimation(fig, update, frames=241, interval=50)
anim.save("resonant_hamiltonian_3D.gif", writer="pillow", fps=20, dpi=150)

plt.tight_layout()
plt.show()