import numpy as np
from matplotlib import pyplot as plt

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


def visualize(true_y, pred_y, t, title=None, savefig=False):
    fig = plt.figure(figsize=(12, 4))
    ax_traj = fig.add_subplot(121)
    ax_phase = fig.add_subplot(122, projection="3d")

    ax_traj.cla()
    ax_traj.set_title('Trajectories')
    ax_traj.set_xlabel('$t$')
    ax_traj.set_ylabel('$u$')
    ax_traj.plot(t, true_y, '-', label="True")
    ax_traj.plot(t, pred_y, '--', label="Pred")
    ax_traj.set_xlim(t.min(), t.max())
    ax_traj.set_ylim(-25, 50)
    ax_traj.legend()
    plt.legend()

    ax_phase.cla()
    ax_phase.set_title('Phase Portrait')
    ax_phase.set_xlabel('$u_1$')
    ax_phase.set_ylabel('$u_2$')
    ax_phase.set_zlabel('$u_3$')
    ax_phase.plot(*true_y.T, 'g-', label="True")
    ax_phase.plot(*pred_y.T, 'b--', label="Pred")
    ax_phase.set_xlim(-25, 25)
    ax_phase.set_ylim(-25, 25)
    ax_phase.set_zlim(0, 50)
    plt.legend()

    fig.tight_layout()
    fig.suptitle(title)
    if savefig:
        plt.savefig(title + '.png')
    else:
        plt.show()

    plt.close()
