'''
@author: Faizan-Uni-Stuttgart

Mar 8, 2021

12:29:19 PM

'''
import os
import sys
import time
import timeit
import traceback as tb
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import rankdata

plt.ioff()

DEBUG_FLAG = True

plt.rcParams.update({'font.size':16})


def main():

    main_dir = Path(os.getcwd())
    os.chdir(main_dir)

#     ser = pd.read_csv(
#         r"P:\Synchronize\IWS\Testings\fourtrans_practice\phsann\inn_daily_discharge.csv",
#         sep=';',
#         index_col=0).loc['1990-01-01':'2000-12-31', '230102']

#     ser = pd.read_csv(
#         r"P:\Synchronize\IWS\Testings\fourtrans_practice\phsann\neckar_norm_cop_infill_discharge_1961_2015_20190118.csv",
#         sep=';',
#         index_col=0).loc['1990-01-01':'2000-12-31', '454']

#     ser = pd.read_csv(
#         r"P:\Synchronize\IWS\Testings\fourtrans_practice\phsann\test_ad_lump_01\sim_data_0.csv",
#         sep=';').loc[:, '427']

#     ser = pd.read_csv(
#         r"P:\Synchronize\IWS\QGIS_Neckar\spate_cmpr__ppt_cosmo__monthly_scale_calib\03_hbv_figs\kf_01_calib_HBV_sim_420.csv",
#         sep=';', index_col=0).loc[:, ['q_obs', 'q_sim']].iloc[365:]

    ser = pd.read_csv(
        r"P:\Synchronize\IWS\Testings\fourtrans_practice\phsann\test_lump_ms_asymm2_01\sim_data_0.csv",
        sep=';').loc[:, ['427', '3421']]

    #==========================================================================
    # Auto coula case
    #==========================================================================
#     lag = 2
#     vals_1 = ser.values.copy()
#     vals_2 = ser.values.copy()
#
#     vals_1 = vals_1[:-lag]
#     vals_2 = vals_2[+lag:]
    #==========================================================================

    #==========================================================================
    # Cross copula case
    #==========================================================================
    vals_1 = ser.values[:, 0]
    vals_2 = ser.values[:, 1]
    #==========================================================================

    probs_1 = rankdata(vals_1) / (vals_1.size + 1)
    probs_2 = rankdata(vals_2) / (vals_2.size + 1)

    plt.figure(figsize=(4, 4))

    plt.scatter(probs_1, probs_2, alpha=0.4, c='k')

    plt.gca().set_aspect('equal')

    plt.grid()
    plt.gca().set_axisbelow(True)

    plt.savefig(
        r"P:\Synchronize\IWS\Projects\2016_DFG_SPATE\Meetings_Second_Phase\20210311_Online\ecop_ms_ad_lump.png",
        dpi=600,
        bbox_inches='tight')

#     plt.show()

    plt.close()
    return


if __name__ == '__main__':
    print('#### Started on %s ####\n' % time.asctime())
    START = timeit.default_timer()

    #==========================================================================
    # When in post_mortem:
    # 1. "where" to show the stack
    # 2. "up" move the stack up to an older frame
    # 3. "down" move the stack down to a newer frame
    # 4. "interact" start an interactive interpreter
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
