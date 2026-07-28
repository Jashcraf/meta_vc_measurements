from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from astropy.io import fits

from katsu.mueller import decompose_depolarizer

# Change to path with data in it
data_pth = Path.home() / "Data/meta_vc_07302026"

# Wavelengths in nm
wavelengths = [
    1480,
    1630,
    1635,
    1650,
    1780,
]

def plot_mueller(M, cmap="RdBu_r", vmin=-1, vmax=1, title=None):

    fig, axs = plt.subplots(ncols=4, nrows=4)

    for i in range(4):
        for j in range(4):

            ax = axs[i, j]
            ax.set_title(f"$M{i}{j}$")

            # Kill axes ticks
            ax.set_xticks([])
            ax.set_xticklabels([])
            ax.set_yticks([])
            ax.set_yticklabels([])

            # Plot the thingy
            im = ax.imshow(M[..., i, j], cmap=cmap, vmin=vmin, vmax=vmax)
            div = make_axes_locatable(ax)
            cax = div.append_axes("right", size="7%", pad="2%")
            fig.colorbar(im, cax=cax)

            if title is not None:
                fig.suptitle(title)


for wvl in wavelengths:

    # load data
    mueller_pth = data_pth / f"spatial_cal_scalar_vortexH_1modes_1e-40ftol_{wvl}nm.fits"
    mueller = fits.getdata(mueller_pth)

    # Plot the Mueller matrix
    plot_mueller(mueller, title=f"{wvl}nm")
    plt.show()
