import numpy as np


def vpt_time(ts, Uts, pre, vpt_tol=5.):
    """
    Valid prediction time for a specific instance.
    """
    def _valid_prediction_index(err, tol):
        """
            First index i where err[i] > tol. err is assumed to be 1D and tol is a float. 
            If err is never greater than tol, then len(err) is returned.
        """
        mask = np.logical_or(err > tol, ~np.isfinite(err))
        if np.any(mask):
            return np.argmax(mask)
        return len(err)

    err = np.linalg.norm((Uts-pre), axis=1, ord=2)
    idx = _valid_prediction_index(err, vpt_tol)
    if idx == 0:
        vptime = 0.
    else:
        vptime = ts[idx-1] - ts[0]
    return vptime