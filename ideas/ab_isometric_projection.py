'''
@author: Faizan-Uni-Stuttgart

Jan 18, 2021

5:10:39 PM

'''
import os
import time
import timeit
from pathlib import Path

# import sympy
import numpy as np
import matplotlib.pyplot as plt

plt.ioff()

DEBUG_FLAG = False


def get_pt_proj(pt):

    # t comes from a plane whose normal is the 1-1 line.

    t = -pt.sum() / pt.size

    pt_proj = pt + t

    return pt_proj


def main():

    main_dir = Path(os.getcwd())
    os.chdir(main_dir)

    pt = np.array([1, 1, 1, 0])

    pt_proj = get_pt_proj(pt)

    dist = ((pt_proj ** 2).sum()) ** 0.5

    print(pt)
    print(pt_proj)
    print(dist)

    return


if __name__ == '__main__':

    _save_log_ = False
    if _save_log_:
        from datetime import datetime
        from std_logger import StdFileLoggerCtrl

        # save all console activity to out_log_file
        out_log_file = os.path.join(
            r'P:\Synchronize\python_script_logs\\%s_log_%s.log' % (
            os.path.basename(__file__),
            datetime.now().strftime('%Y%m%d%H%M%S')))

        log_link = StdFileLoggerCtrl(out_log_file)

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
            import pdb
            pdb.post_mortem()

    else:
        main()

    STOP = timeit.default_timer()
    print(('\n#### Done with everything on %s.\nTotal run time was'
           ' about %0.4f seconds ####' % (time.asctime(), STOP - START)))

    if _save_log_:
        log_link.stop()
