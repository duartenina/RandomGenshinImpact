import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.optimize import curve_fit
from scipy.stats import chisquare

plt.style.use('default')

# residuals_f = lambda func, xdata, ydata, pars: ydata - func(xdata, *pars)
# residual_sum_sq_f = lambda residuals: np.sum(residuals ** 2)
# chi_squared_f = lambda residual_sum_sq, N, n: residual_sum_sq / (N - n)
# r_sq_f = lambda residual_sum_sq, ydata: (
#     1 - (residual_sum_sq / np.sum((ydata - np.mean(ydata)) ** 2))
# )
# par_errors_f = lambda cov_mat: np.sqrt(np.diag(cov_mat))

r_sq_f = lambda ydata, fitdata: (1 - (
    np.sum((ydata - fitdata) ** 2)
    /
    np.sum((ydata - np.mean(ydata)) ** 2)
))
# chi_sq_f = lambda ydata, fitdata, n_pars: (
#     np.sum((ydata - fitdata) ** 2 / fitdata)# / (len(ydata) - n_pars)
# )

exp_data = np.loadtxt('data/char_exp.csv', delimiter=',')

lvl, exp, exp_total = np.split(exp_data, 3, axis=1)
lvl, exp, exp_total = lvl.flatten(), exp.flatten(), exp_total.flatten()
exp_total = np.cumsum(exp)

# exp from 1-80
fits_info = {
    'EXP 1-80 / Poly 2nd deg': (
        exp, (lvl < 81), lambda x, a, b, c: np.polyval((a, b, c), x),
        dict(color='cyan', ls='-')
    ),
    'EXP 81-90 / Poly 2nd deg': (
        exp, (lvl > 80), lambda x, a, b, c: np.polyval((a, b, c), x),
        dict(color='lightblue', ls='-')
    ),
    # 'EXP 81-90 / Exp': (
    #     exp, (lvl > 80), lambda x, a, b: a * np.exp(b*x),
    #     dict(color='green', ls='-')
    # ),
    'Cum. EXP 1-80 / Poly 3rd deg': (
        exp_total, (lvl < 81), lambda x, a, b, c, d: np.polyval((a, b, c, d), x),
        dict(color='red', ls='-')
    ),
    'Cum. EXP 81-90 / Poly 3rd deg': (
        exp_total, (lvl > 80), lambda x, a, b, c, d: np.polyval((a, b, c, d), x),
        dict(color='orange', ls='-')
    ),
}
fits = {}
for fit_name in fits_info:
    y, inds, fit_func, _ = fits_info[fit_name]
    pars, pcov = curve_fit(fit_func, lvl[inds], y[inds])
    fits[fit_name] = (pars, pcov)


plot_funcs = (plt.plot, plt.plot, plt.plot, plt.plot, plt.semilogy)
plt.figure(figsize=(10, 15))
gs = GridSpec(len(plot_funcs), 1, hspace=0, height_ratios=(1, .5, .5, .5, 1))

################################################
for n, plot_func in enumerate(plot_funcs):
    ax = plt.subplot(gs[n])

    if n < len(plot_funcs) - 1:
        fac = 1e-6
    else:
        fac = 1

    plt.axvline(80.5, color='k', ls='--')

    plot_func(lvl, exp*fac, '.', ms=10, color='darkblue', label='EXP')
    plot_func(lvl, exp_total*fac, '.', ms=10, color='brown', label='Cumulative EXP')

    for fit_name in fits_info:
        ys, inds, fit_func, style = fits_info[fit_name]
        pars, pcov = fits[fit_name]

        y, fity = ys[inds], fit_func(lvl[inds], *pars)
        r_sq = r_sq_f(y, fity)
        chi_sq = chisquare(y, fity)[0] / (len(y) - len(pars))

        plot_func(lvl, fit_func(lvl, *pars)*fac, **style, label=(
            fit_name +
            f' ($R^2$ = {r_sq:.3f}; ' +
            '$\\chi^2_{red}$ = ' + f'{chi_sq:.3f})'
        ))

    if n == 0:
        ax.xaxis.tick_top()
    elif n == len(plot_funcs) - 1:
        plt.xlabel('LVL')

    if fac == 1:
        plt.ylabel('EXP')
    else:
        plt.ylabel('EXP ($\\times 10^{6}$)')

    if n == 0:
        plt.ylim(-0.5, 10)
    elif n == 1:
        plt.ylim(-0.01, 0.04)
    elif n == 2:
        plt.ylim(0.1, 0.4)
    elif n == 3:
        plt.ylim(3, 7)
    elif n == 4:
        plt.ylim(5e2, 2e7)

    if n == 0:
        plt.legend(loc='upper left')

plt.tight_layout()

plt.savefig('images/char_exp.png')
plt.show()