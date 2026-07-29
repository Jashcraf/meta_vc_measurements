from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from astropy.io import fits

from katsu.mueller import decompose_depolarizer, decompose_retarder

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

rms_dia = []
rms_ret = []

def plot_mueller(M, cmap="RdBu_r", vmin=-1, vmax=1, title=None):

    fig, axs = plt.subplots(ncols=4, nrows=4, figsize=[10, 10])

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

def clean_array(M):
    M[np.isnan(M)] = 0.
    M[np.isinf(M)] = 0.
    M[M == "NaN"] = 0.
    M[M == np.nan] = 0.
    return M

def valid_mask(M):
    """Pixels that permit a Lu-Chipman decomposition."""
    T = M[..., 0, 0]
    with np.errstate(invalid="ignore", divide="ignore"):
        D = np.sqrt(np.sum((M[..., 0, 1:] / T[..., None])**2, axis=-1))
    return np.all(np.isfinite(M), axis=(-2, -1)) & (T > 0) & (D < 1)


for wvl in wavelengths:

    # load data
    mueller_pth = data_pth / f"spatial_cal_scalar_vortexH_1modes_1e-40ftol_{wvl}nm.fits"
    mueller = fits.getdata(mueller_pth)
    good = valid_mask(mueller)
    
    M_dia = np.full_like(mueller, np.nan)
    M_ret = np.full_like(mueller, np.nan)
    M_dep = np.full_like(mueller, np.nan)

            
    # Break up the mueller matrix
    M_dep[good], M_ret[good], M_dia[good] = decompose_depolarizer(mueller[good], return_all=True)

    plot_mueller(M_ret)
    plt.show()

    mse_dia = np.nanmean((M_dia - np.eye(4))**2)
    mse_ret = np.nanmean((M_ret - np.eye(4))**2)
    rms_dia.append(mse_dia)
    rms_ret.append(mse_ret)

plt.figure()
plt.plot(wavelengths, rms_dia, label="Diattenuation", marker="x", linestyle="None")
plt.plot(wavelengths, rms_ret, label="Retardance", marker="o", linestyle="None")
plt.ylabel("Mean Squared Error to Identity Matrix")
plt.xlabel("Wavelength, nm")
plt.legend()
plt.show()

