"""
Visualise slope_loss across the 4-parameter space (OmegaM, OmegaK, Omegab, h).
Data are stored with Omegab_ratio = Omegab/OmegaM; convert to Omegab on load.

Upper triangle: griddata contour combining master_log scatter (broad coverage) and
                the pre-computed 10x10 grid (fine resolution in low-loss region)
Diagonal:       1-D profile histogram
Lower triangle: scatter coloured by log10(slope_loss)
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.ticker import MaxNLocator
from scipy.interpolate import griddata


# ── paths ─────────────────────────────────────────────────────────────────────
grid_folder       = './data/grid_losses'
fix2params_folder = './data/fix2params'
fix3params_folder = './data/fix3params'
losses_1d_folder  = './data/1d_losses'

# ── load data ────────────────────────────────────────────────────────────────
# log format: total_loss slope_loss per_k_integer_loss local_chi2 param0..3 <k_array...>
data = []
with open('../data/try_intK_planck_optuna/master_log.txt') as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) < 8:
            continue
        data.append([float(x) for x in parts[:8]])

arr = np.array(data)
slope_loss = arr[:, 1]
params     = arr[:, 4:8]   # columns: OmegaM, OmegaK, Omegab_ratio, h

# convert Omegab_ratio → Omegab  (col 2 = Omegab_ratio, col 0 = OmegaM)
params[:, 2] = params[:, 2] * params[:, 0]

param_names = [r'$\Omega_m$', r'$\Omega_K$', r'$\Omega_b$', r'$h$']

log_sl = np.log10(slope_loss)
vmin, vmax = np.percentile(log_sl, 2), np.percentile(log_sl, 98)

# contour levels: exponentially spaced in log_sl so bands are denser at low slope_loss
_t = np.linspace(0, 1, 21)
contour_levels = vmin + (np.exp(_t) - 1) / (np.e - 1) * (vmax - vmin)

# chosen best-fit point (stored as OmegaM, OmegaK, Omegab_ratio, h)
# _best_raw = np.array([0.4444730086581741, -0.03940583926181416, 0.16336760287147203, 0.5615471381223595]) # cosmo_param_direc
_best_raw = np.array([0.4444730086581741, -0.03940583926181416, 0.16169213605504282, 0.5615471381223595]) 
# _best_raw = np.array([0.45531269806747615, -0.0397827163256654, 0.15804679013016237, 0.5647692899377309]) # cosmo_param_direc_OmegabStep
best_params = _best_raw.copy()
best_params[2] = _best_raw[2] * _best_raw[0]   # Omegab = Omegab_ratio * OmegaM

# parameter ranges: half the full span, centred on best-fit point
# (computed after conversion so Omegab range is in physical units)
p_ranges = []
for i in range(4):
    span = params[:, i].max() - params[:, i].min()
    half = span / 4
    p_ranges.append((best_params[i] - half, best_params[i] + half))

# ── load fix3params data (one varying param, three fixed at best-fit) ────────
_pname_1d = ['OmegaM', 'OmegaK', 'Omegab_ratio', 'h']
fix3params_data = {}
for _i, _nm in enumerate(_pname_1d):
    fpath = os.path.join(fix3params_folder, f'log_{_nm}.txt')
    if os.path.exists(fpath):
        _d = []
        with open(fpath) as _f:
            for line in _f:
                parts = line.strip().split()
                if len(parts) >= 8:
                    _d.append([float(x) for x in parts[:8]])
        if _d:
            d = np.array(_d)
            d[:, 6] = d[:, 6] * d[:, 4]   # Omegab_ratio → Omegab
            fix3params_data[_i] = d

# ── load fix2params data ──────────────────────────────────────────────────────
_pname = ['OmegaM', 'OmegaK', 'Omegab_ratio', 'h']
fix2params_data = {}
for _r in range(4):
    for _c in range(_r + 1, 4):
        fpath = os.path.join(fix2params_folder,
                             f'log_{_pname[_r]}_{_pname[_c]}.txt')
        if os.path.exists(fpath):
            _d = []
            with open(fpath) as _f:
                for line in _f:
                    parts = line.strip().split()
                    if len(parts) >= 8:
                        _d.append([float(x) for x in parts[:8]])
            if _d:
                d = np.array(_d)
                # convert Omegab_ratio (col 4+2) → Omegab using OmegaM (col 4+0)
                d[:, 4 + 2] = d[:, 4 + 2] * d[:, 4 + 0]
                fix2params_data[(_r, _c)] = d

# ── figure ───────────────────────────────────────────────────────────────────
n = 4
cmap = 'viridis_r'

fig, axes = plt.subplots(n, n, figsize=(10, 10),
                         gridspec_kw={'wspace': 0.06, 'hspace': 0.06})

for row in range(n):
    for col in range(n):
        ax = axes[row, col]

        if row == col:
            # ── diagonal: log10(slope_loss) vs parameter ───────────────────
            i = col
            xs_d, ys_d = [], []

            # # source 1: uniform 1-D scan (compute_1D_losses.py)
            # npz_file = os.path.join(losses_1d_folder, f'1d_param{i}.npz')
            # if os.path.exists(npz_file):
            #     d1d  = np.load(npz_file, allow_pickle=True)
            #     xi_1 = d1d['xi'].copy()
            #     sl_1 = d1d['slope_loss'].copy()
            #     if i == 2:
            #         xi_1 = xi_1 * _best_raw[0]   # Omegab_ratio → Omegab
            #     f1 = np.isfinite(sl_1) & (sl_1 > 0)
            #     xs_d.append(xi_1[f1])
            #     ys_d.append(np.log10(sl_1[f1]))

            # source 2: Optuna TPE scatter (calculate_integer_loss_func_fix3params_optuna.py)
            if i in fix3params_data:
                d3   = fix3params_data[i]
                xi_3 = d3[:, 4 + i]
                sl_3 = d3[:, 1]
                f3   = np.isfinite(sl_3) & (sl_3 > 0)
                xs_d.append(xi_3[f3])
                ys_d.append(np.log10(sl_3[f3]))

            if xs_d:
                x_all = np.concatenate(xs_d)
                y_all = np.concatenate(ys_d)
                sort_idx = np.argsort(x_all)                                                                                           
                ax.scatter(x_all, y_all, s=6, color='steelblue', alpha=0.6, zorder=2)
                ax.plot(x_all[sort_idx], y_all[sort_idx],
                        color='navy', lw=1.0, zorder=3)

            ax.axvline(best_params[i], color='red', lw=1.5, ls='--')
            ax.set_xlim(p_ranges[i])
            ax.xaxis.set_major_locator(MaxNLocator(nbins=4, prune='both'))
            ax.yaxis.set_major_locator(MaxNLocator(nbins=4, prune='both'))
            ax.tick_params(axis='y', labelsize=9)

        elif row < col:
            # ── upper triangle: contour from fix2params + pre-computed grid ──
            xs, ys, zs = [], [], []

            # source 1: fix2params scatter
            key = (row, col)
            if key in fix2params_data:
                d2 = fix2params_data[key]
                xs.append(d2[:, 4 + col])
                ys.append(d2[:, 4 + row])
                zs.append(np.log10(d2[:, 1]))

            # source 2: pre-computed 10x10 grid
            grid_file = os.path.join(grid_folder, f'grid_r{row}_c{col}.npz')
            if os.path.exists(grid_file):
                gd     = np.load(grid_file)
                sl_raw = gd['slope_loss'].copy()
                sl_raw[~np.isfinite(sl_raw)] = np.nan
                # xi → param[col], yi → param[row]
                Xi_g, Yi_g = np.meshgrid(gd['xi'], gd['yi'])

                # convert grid axes if either axis is Omegab_ratio (index 2)
                if col == 2:
                    # xi = Omegab_ratio; multiply by OmegaM
                    OmegaM_for_x = Yi_g if row == 0 else best_params[0]
                    Xi_g = Xi_g * OmegaM_for_x
                if row == 2:
                    # yi = Omegab_ratio; multiply by OmegaM
                    OmegaM_for_y = Xi_g if col == 0 else best_params[0]
                    Yi_g = Yi_g * OmegaM_for_y

                mask = np.isfinite(sl_raw)
                xs.append(Xi_g[mask])
                ys.append(Yi_g[mask])
                zs.append(np.log10(sl_raw[mask]))

            if xs:
                x_all = np.concatenate(xs)
                y_all = np.concatenate(ys)
                z_all = np.concatenate(zs)
                xi2   = np.linspace(x_all.min(), x_all.max(), 60)
                yi2   = np.linspace(y_all.min(), y_all.max(), 60)
                Xi2, Yi2 = np.meshgrid(xi2, yi2)
                Zi   = griddata((x_all, y_all), z_all, (Xi2, Yi2), method='linear')
                Zi   = np.ma.masked_invalid(Zi)
                ax.contourf(Xi2, Yi2, Zi, levels=contour_levels,
                            cmap=cmap, vmin=vmin, vmax=vmax, extend='both')

            ax.scatter(best_params[col], best_params[row],
                       marker='*', s=150, color='red', zorder=5)
            ax.set_xlim(p_ranges[col])
            ax.set_ylim(p_ranges[row])
            ax.xaxis.set_major_locator(MaxNLocator(nbins=4, prune='both'))
            ax.yaxis.set_major_locator(MaxNLocator(nbins=4, prune='both'))

        else:
            # ── lower triangle: scatter coloured by log(slope_loss) ─────────
            ax.scatter(params[:, col], params[:, row],
                       c=log_sl, cmap=cmap, vmin=vmin, vmax=vmax,
                       s=4, alpha=0.7, rasterized=True)
            ax.scatter(best_params[col], best_params[row],
                       marker='*', s=150, color='red', zorder=5)
            ax.set_xlim(p_ranges[col])
            ax.set_ylim(p_ranges[row])
            ax.xaxis.set_major_locator(MaxNLocator(nbins=4, prune='both'))
            ax.yaxis.set_major_locator(MaxNLocator(nbins=4, prune='both'))

        # ── tick label visibility: outer edges only ─────────────────────────
        if row == n - 1:
            ax.tick_params(axis='x', labelsize=10, rotation=30)
        else:
            ax.tick_params(axis='x', labelbottom=False)

        if col == 0:
            ax.tick_params(axis='y', labelsize=9 if row == col else 10)
        else:
            ax.tick_params(axis='y', labelleft=False)

# ── axis labels on outer edges ────────────────────────────────────────────────
for col in range(n):
    axes[-1, col].set_xlabel(param_names[col], fontsize=16, labelpad=8)
axes[0, 0].set_ylabel(r'$\log_{10}(\mathrm{slope\_loss})$', fontsize=10, labelpad=4)
for row in range(1, n):
    axes[row, 0].set_ylabel(param_names[row], fontsize=16, labelpad=8)

# ── colorbar ─────────────────────────────────────────────────────────────────
sm = plt.cm.ScalarMappable(cmap=cmap, norm=mcolors.Normalize(vmin=vmin, vmax=vmax))
sm.set_array([])
cbar_ax = fig.add_axes([0.93, 0.08, 0.015, 0.85])
fig.colorbar(sm, cax=cbar_ax, label=r'$\log_{10}(\mathrm{slope\_loss})$')

plt.suptitle('slope_loss landscape across cosmological parameter space',
             y=1.01, fontsize=13)

out = './figures/loss_param_space_Omegab.pdf'
plt.savefig(out, bbox_inches='tight', dpi=150)
print(f'Saved: {out}')

# ── supplementary: 1-D marginals for slope_loss ──────────────────────────────
fig2, axs = plt.subplots(1, 4, figsize=(14, 3.5))
low_sl = slope_loss < np.percentile(slope_loss, 10)

for i, (ax, name) in enumerate(zip(axs, param_names)):
    ax.hist(params[:, i], bins=30, color='steelblue', alpha=0.5, label='all')
    ax.hist(params[low_sl, i], bins=20, color='crimson', alpha=0.7,
            label='best 10%\n(slope_loss low)')
    ax.set_xlabel(name, fontsize=16)
    ax.set_ylabel('count', fontsize=13)
    ax.tick_params(axis='both', labelsize=11)
    ax.legend(fontsize=8)

plt.suptitle('Parameter distribution: all trials vs. lowest slope_loss region', fontsize=12)
plt.tight_layout()

out2 = './figures/loss_param_marginals_Omegab.pdf'
plt.savefig(out2, bbox_inches='tight', dpi=150)
print(f'Saved: {out2}')
