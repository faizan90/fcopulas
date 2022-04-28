'''
@author: Faizan-Uni-Stuttgart

Mar 2, 2022

2:55:26 PM

'''
import os
import sys
import time
import timeit
import traceback as tb
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt; plt.ioff()

DEBUG_FLAG = False


def main():

    main_dir = Path(r'P:\Synchronize\IWS\Testings\fourtrans_practice\phsann')
    os.chdir(main_dir)

    in_file_path = Path(r'BW_dwd_stns_60min_1995_2020_data.csv')

    labels = [
        'P00071', 'P00257', 'P00279', 'P00498', 'P00684', 'P00757', 'P00931',
        'P01089', 'P01216', 'P01224', 'P01255', 'P01290', 'P01584', 'P01602',
        'P01711', 'P01937', 'P02388', 'P02575', 'P02638', 'P02787', 'P02814',
        'P02880', 'P03278', 'P03362', 'P03519', 'P03761', 'P03925', 'P03927',
        'P04160', 'P04175', 'P04294', 'P04300', 'P04315', 'P04349', 'P04623',
        'P04710', 'P04881', 'P04928', 'P05229', 'P05664', 'P05711', 'P05724',
        'P05731', 'P06258', 'P06263', 'P06275', 'P07138', 'P07187', 'P07331',
        'P13672', 'P13698', 'P13965']

    time_fmt = '%Y-%m-%dT%H:%M:%S'

    beg_time = '2010-01-01'
    end_time = '2014-12-31'

    sep = ';'

    corr_type = 'spearman'

    min_str_len = 6
    #==========================================================================

    if in_file_path.suffix == '.csv':
        in_df = pd.read_csv(in_file_path, sep=sep, index_col=0)
        in_df.index = pd.to_datetime(in_df.index, format=time_fmt)

    elif in_file_path.suffix == '.pkl':
        in_df = pd.read_pickle(in_file_path)

    else:
        raise NotImplementedError(
            f'Unknown extension of input file: '
            f'{in_file_path.suffix}!')

    in_df = in_df.loc[beg_time:end_time, labels]
    #==========================================================================

    corrs_df = in_df.corr(corr_type).round(3)

    print(corrs_df)
    #==========================================================================

    if False:
        take_idxs = np.where(
            np.triu(np.ones_like(corrs_df.values), k=1).ravel())

        corrs_dist = corrs_df.values.ravel()[take_idxs]

        corrs_dist.sort()

        probs = np.arange(1.0, corrs_dist.size + 1.0) / (corrs_dist.size + 1.0)

        plt.plot(corrs_dist, probs, c='k', alpha=0.75)

        plt.grid()
        plt.gca().set_axisbelow(True)

        plt.xlabel('Correlation')
        plt.ylabel('Non-exceedence probability')

        plt.title(f'Distribution of {corr_type} correlation for all stations')

        plt.show()
        plt.close()
    #==========================================================================

    if False:
        with open('nebors_stn_names_corrs.csv', 'w') as name_hdl:
            for ref_stn in corrs_df.index:
                stn_corrs_ser = corrs_df.loc[ref_stn].copy()

                stn_corrs_ser.sort_values(inplace=True, ascending=False)

                neb_str = []
                corr_str = []
                for neb_stn in stn_corrs_ser.index:
                    neb_corr = stn_corrs_ser.loc[neb_stn]

                    len_neb_stn = len(neb_stn)

                    if len_neb_stn < min_str_len:
                        neb_stn = (
                            ' ' * (min_str_len - len_neb_stn)) + neb_stn

                        len_neb_stn = min_str_len

                    neb_str.append(neb_stn)
                    corr_str.append(f'{neb_corr:+{len_neb_stn}.3f}')

                name_hdl.write(';'.join(neb_str))
                name_hdl.write('\n')

                name_hdl.write(';'.join(corr_str))
                name_hdl.write('\n')

    return


if __name__ == '__main__':
    print('#### Started on %s ####\n' % time.asctime())
    START = timeit.default_timer()

    #==========================================================================
    # When in post_mortem:
    # 1. "where" to show the stack,
    # 2. "up" move the stack up to an older frame,
    # 3. "down" move the stack down to a newer frame, and
    # 4. "interact" start an interactive interpreter.
    #==========================================================================

    if DEBUG_FLAG:
        try:
            main()

        except:
            pre_stack = tb.format_stack()[:-1]

            err_tb = list(tb.TracebackException(*sys.exc_info()).format())

            lines = [err_tb[0]] + pre_stack + err_tb[2:]

            for line in lines:
                print(line, file=sys.stderr, end='')

            import pdb
            pdb.post_mortem()
    else:
        main()

    STOP = timeit.default_timer()
    print(('\n#### Done with everything on %s.\nTotal run time was'
           ' about %0.4f seconds ####' % (time.asctime(), STOP - START)))
