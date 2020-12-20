from matplotlib.pyplot import bar, xlabel, ylabel
import numpy as np
import matplotlib.pyplot as plt
from numpy.core.numeric import isfortran
import pandas as pd
from pandas.core.reshape.concat import concat


banner_types = ['char', 'novice', 'permanent', 'weapon']
wish_columns = ['type', 'name', 'rarity', 'date']
wish_dtype = ['string', 'string', 'string', 'datetime']
wish_dtype = ['int']*4
raw_filename = lambda nick, bar_label: f'data/{nick}/{bar_label}_raw.dat'
clean_filename = lambda nick, bar_label: f'data/{nick}/{bar_label}_clean.csv'
graphs_folder = 'images/'
# colours = plt.rcParams['axes.prop_cycle'].by_key()['color']
cmap = plt.get_cmap('Set3')
colours = cmap(np.linspace(0, 1, 12))
rarity_colours = ['white', 'grey', 'green', 'blue', 'magenta', 'gold']
cell_colour = (0.1, 0.1, 0.1)
header_colour = (0.2, 0.2, 0.2)
banner_info = {
    'Venti':  (('2020-09-28', '2020-10-18'), 'green'),
    'Klee':   (('2020-10-20', '2020-11-09'), 'red'),
    'Childe': (('2020-11-11', '2020-11-30'), 'blue'),
    'Zhongli': (('2020-12-01', '2021-01-30'), 'brown'),
}


def convert_raw_to_table(bar_label, nick='drnbw'):
    wishes = []
    with open(raw_filename(nick, bar_label), 'r') as in_file:
        wishes = in_file.read().split('\n')

    wishes_table = []
    new_wish = []
    for n, wish_part in enumerate(wishes):
        if n%3 == 1:
            name_and_rarity = wish_part.split(' (')

            name = name_and_rarity[0]
            if len(name_and_rarity) > 1:
                rarity = name_and_rarity[1].replace(')', '')
            else:
                rarity = '3-Star'

            wishes[n] = name

            new_wish += [name, rarity]
        else:
            new_wish.append(wish_part)

        if n%3 == 2:
            wishes_table.append(new_wish)
            new_wish = []

    wishes_clean = pd.DataFrame(
        wishes_table[::-1],
        columns=wish_columns
    )

    wishes_clean['date'] = pd.to_datetime(wishes_clean['date'])

    # wishes_clean.sort_values('date', inplace=True)

    with open(clean_filename(nick, bar_label), 'w') as out_file:
        wishes_clean.to_csv(out_file, index=False, line_terminator='\n')

    return wishes_clean


def concat_all_wishes(nick='drnbw'):
    wishes_dict = {
        bar_label: convert_raw_to_table(bar_label, nick=nick)
        for bar_label in banner_types
    }

    to_concat = []
    for bar_label in banner_types:
        temp = wishes_dict[bar_label].copy()
        temp['banner'] = bar_label
        to_concat.append(temp)

    wishes_all = pd.concat(to_concat)

    with open(f'data/{nick}/wishes_all.csv', 'w') as f:
        wishes_all.sort_values('date').to_csv(f, index=False, line_terminator='\n')

    return wishes_all, wishes_dict


def plot_hist_of_by_(wishes_all, bin_column, bar_column,
                     file_suffix='', title_suffix='', nick='drnbw'):
    plt.figure(figsize=(10, 5))

    xlabels = pd.unique(wishes_all[bin_column])
    zlabels = [
        type
        for type, data in wishes_all.groupby(bar_column, as_index=False)
    ]

    num_bins = len(xlabels)
    num_bars = len(zlabels)

    xinds = np.arange(num_bins)

    plt.subplot(121)

    data_grouped = wishes_all.groupby(bar_column, as_index=False)
    bottom = np.zeros(num_bins)
    table_text = []
    cell_colours = []
    totals = np.zeros(num_bins + 1)
    for n, (bar_label, data) in enumerate(data_grouped):
        bar_heights = [np.sum(data[bin_column] == x) for x in xlabels]

        plt.bar(xinds, bar_heights, bottom=bottom,
                width=0.5,
                color=colours[n],
                label=bar_label)
        bottom += bar_heights

        total = np.sum(bar_heights)

        table_text.append(
            [bar_label] +
            [f'{h:.0f}' for h in bar_heights] +
            [f'{total:.0f}']
        )
        cell_colours.append(
            [colours[n]] + [cell_colour]*(num_bins + 1)
        )

        totals += bar_heights + [total]

    plt.legend()

    plt.xticks(xinds, xlabels)
    plt.xlabel(bin_column)
    plt.ylabel('# Occurrences')

    # Table

    plt.subplot(122)
    plt.axis('off')

    table_text = table_text[::-1]
    table_text.insert(0, [''] + list(xlabels) + ['Total'])
    table_text.append(['Total'] + [f'{t:.0f}' for t in totals])

    num_rows = len(table_text)
    num_cols = len(table_text[0])

    cell_colours = cell_colours[::-1]
    cell_colours.insert(0, [header_colour]*num_cols)
    cell_colours.append([header_colour] + [cell_colour]*(num_cols - 1))

    table = plt.table(
        table_text,
        cellColours=cell_colours,
        # colLabels=list(xlabels) + ['Total'],
        # colColours=[header_colour]*num_cols,
        # rowLabels=zlabels[::-1] + ['Total'],
        # rowColours=colours[:num_bars][::-1] + [header_colour],
        bbox=[0., 0., 1.0, 1.0]
    )

    for n in range(num_rows):
        if n < num_bars:
            table[(n+1, 0)].get_text().set_color('k')
            # table[(n, 0)].set_backgroundcolor(colours[n])

    for n in range(num_cols):
        table.auto_set_column_width(n)

    title = f'{nick} - Distribution of {bin_column} by {bar_column}'
    if title_suffix != '':
        title += f' ({title_suffix})'

    plt.suptitle(title, y=0.975)

    plt.tight_layout(rect=(0,0,1,0.95))

    plt.savefig(
        graphs_folder +
        f'{nick}_distr_of_{bin_column}_by_{bar_column}{file_suffix}',
        dpi=100
    )

    plt.show()


def plot_wishes_in_time(wishes_all, nick='drnbw', banner='char'):
    wishes_banner = wishes_all[wishes_all['banner'] == banner]
    rarities = wishes_banner['rarity'].apply(
        lambda c: int(c.split('-')[0])
    )

    last_wish = len(wishes_banner) - 1
    min_ind, max_ind = -last_wish*0.02, last_wish*1.05

    plt.figure(figsize=(8, 6))

    for banner_char in banner_info:
        banner_startend, color = banner_info[banner_char]
        start, end = np.array(banner_startend, dtype='datetime64')

        after_start = wishes_banner['date'].dt.date >= start
        before_end = wishes_banner['date'].dt.date <= end

        if not np.any(after_start) or not np.any(before_end):
            continue

        if np.all(after_start):
            start_ind = min_ind
        else:
            start_ind = np.argmax(after_start)

        if np.all(before_end):
            end_ind = max_ind
        else:
            end_ind = np.argmin(before_end) - 1

        plt.axvspan(start_ind, end_ind, color=color, alpha=0.2, hatch='/')
        plt.text((start_ind + end_ind)/2, 4.5, f'{banner_char} banner',
                 va='center', ha='center')

    plt.axvline(0, ls=':', color=(.7, .7, .7, .7))
    plt.axvline(last_wish, ls=':', color=(.7, .7, .7, .7))

    plt.text(last_wish*1.02, 4, s=f'last wish ({len(wishes_banner)} so far)',
             rotation=90, ha='left', va='center')

    for rarity in [3,4,5]:
        rarity_text = f'{rarity}-Star'
        color = rarity_colours[rarity]

        is_of_rarity = wishes_banner['rarity'] == rarity_text
        is_weapon = wishes_banner['type'] == 'Weapon'

        inds_weapon = wishes_banner[is_of_rarity & is_weapon].index
        inds_char = wishes_banner[is_of_rarity & ~is_weapon].index
        inds = wishes_banner[is_of_rarity].index

        plt.plot(inds_weapon, rarities[inds_weapon], 'o',
                 color=color, ms=rarity)
        plt.plot(inds_char, rarities[inds_char], 'd',
                 color=color, ms=5)

        if rarity != 3:
            since_star = last_wish - inds.max()

            inds_complete = np.hstack((0, inds, last_wish))
            for n, ind in enumerate(inds_complete[:-1]):
                x0 = ind
                x1 = inds_complete[n+1]
                dx = x1 - x0
                xm = x0 + dx/2
                spacing = 0

                plt.text(xm, rarity + (n%2 -.5) * 0.15,
                         f'{dx}',
                         color=color,
                         ha='center', va='center')
                if (dx - spacing) > last_wish/40:
                    plt.annotate(
                        '',
                        (x0 + spacing, rarity),
                        (x1 - spacing, rarity),
                        arrowprops=dict(arrowstyle='<->', color=color)
                    )

    plt.plot(np.nan, 'ow', label='Weapon')
    plt.plot(np.nan, 'dw', label='Character')
    plt.legend(ncol=2, loc='lower center')

    plt.xlim(min_ind, max_ind)
    plt.ylim(2.75, 5.2)
    plt.yticks((3,4,5), (f'{d}-Star' for d in range(3,6)))

    plt.title(f'Wishes in time of {banner} banner for {nick}')

    plt.tight_layout()

    plt.savefig(
        graphs_folder +
        f'{nick}_wishes_in_time_{banner}',
        dpi=100
    )

    plt.show()


def plot_pull_chance_histogram(wishes_all, nick='drnbw'):
    star_4_inds = wishes_all[wishes_all['rarity'] == '4-Star'].index
    space_4_star = np.diff(star_4_inds)

    plt.figure()

    x, y, _ = plt.hist(space_4_star, bins=np.arange(0.5, 11.5, 1),
                       rwidth=0.5, align='mid')

    plt.show()


def do_all_for_nick(wishes_all, nick='drnbw'):
    plot_hist_of_by_(wishes_all, 'rarity', 'banner', nick=nick)
    plot_hist_of_by_(wishes_all, 'rarity', 'type', nick=nick)
    plot_hist_of_by_(wishes_all, 'banner', 'type', nick=nick)
    plot_hist_of_by_(wishes_all[wishes_all['rarity'] == '4-Star'],
                    'banner', 'type', nick=nick,
                    file_suffix='_4_stars',
                    title_suffix='only 4 Stars')
    plot_hist_of_by_(wishes_all[wishes_all['rarity'] == '5-Star'],
                    'banner', 'type', nick=nick,
                    file_suffix='_5_stars',
                    title_suffix='only 5 Stars')

    for banner in banner_types:
        plot_wishes_in_time(
            wishes_all, nick=nick, banner=banner
        )


nick = 'drnbw'
nick = 'samael'
nick = 'sophie'
wishes_all, wishes_dict = concat_all_wishes(nick=nick)
do_all_for_nick(wishes_all, nick=nick)
# plot_wishes_in_time(wishes_all, nick=nick, banner='char')
