import pandapower as pp
import pandapower.networks as ppnw
import numpy as np


np.set_printoptions(
    formatter={
        "complexfloat": lambda x: "{0:.3f}".format(x),
        "float_kind": lambda x: "{0:.3f}".format(x),
    }
)


def init_v(net, n, pd2ppc):
    """Initial bus voltage vector using generator setpoint or 0j+1pu."""
    v = [0j + 1 for _ in range(n)]
    for r in net.gen.itertuples():
        v[pd2ppc[r.bus]] = r.vm_pu
    for r in net.ext_grid.itertuples():
        v[pd2ppc[r.bus]] = r.vm_pu * np.exp(1j * r.va_degree * np.pi / 180)
    return np.array(v, dtype=np.complex64)


def scheduled_p_q(net, n, pd2ppc):
    """Return known per unit real and reactive power injected at each bus.
    Does not include slack real/reactive powers nor PV gen reactive power.
    """
    psch, qsch = {b: 0 for b in range(n)}, {b: 0 for b in range(n)}
    for r in net.gen.itertuples():
        psch[pd2ppc[r.bus]] += r.p_mw / net.sn_mva
    for r in net.sgen.itertuples():
        psch[pd2ppc[r.bus]] += r.p_mw / net.sn_mva
        qsch[pd2ppc[r.bus]] += r.q_mvar / net.sn_mva
    for r in net.load.itertuples():
        psch[pd2ppc[r.bus]] -= r.p_mw / net.sn_mva
        qsch[pd2ppc[r.bus]] -= r.q_mvar / net.sn_mva
    return psch, qsch


def run_lf(net, tol=1e-9, comp_tol=1e-3, max_iter=10000):
    """Perform NR power flow on the given pandapower network.

    Args:
        net (pp.Network): Solved Pandapower network object that defines the
            elements of the network and contains the ybus matrix.
        tol (float): Convergence tolerance (voltage).
        comp_tol (float): Tolerance for comparison check against pandapower.
        max_iter(int): Max iterations to solve load flow.

    Returns:
        float: Sum of real power injected by slack buses in the network.
    """
    ybus = np.array(net._ppc["internal"]["Ybus"].todense())
    pd2ppc = net._pd2ppc_lookups["bus"]  # Pandas bus num --> internal bus num.
    n = ybus.shape[0]  # Number of buses.
    slack_buses = set(pd2ppc[net.ext_grid["bus"]])
    gen_buses = set([pd2ppc[b] for b in net.gen["bus"]])
    ybus_hollow = ybus * (1 - np.eye(n))  # ybus with diagonal elements zeroed.
    v = init_v(net, n, pd2ppc)
    psch, qsch = scheduled_p_q(net, n, pd2ppc)

    it = 0
    while it < max_iter:
        old_v, v = v, [x for x in v]
        for b in [b for b in range(n) if b not in slack_buses]:
            qsch_b = (
                -1 * np.imag(np.conj(old_v[b]) * np.sum(ybus[b, :] * old_v))
                if b in gen_buses
                else qsch[b]
            )
            v[b] = (1 / ybus[b, b]) * (
                (psch[b] - 1j * qsch_b) / np.conj(old_v[b])
                - np.sum(ybus_hollow[b, :] * old_v)
            )
            if b in gen_buses:
                v[b] = np.abs(old_v[b]) * v[b] / np.abs(v[b])  # Only use angle.
        it += 1
        v = np.array(v)
        if np.allclose(v, old_v, rtol=tol, atol=0):
            break
    p_slack = sum(
        (np.real(np.conj(v[b]) * np.sum(ybus[b, :] * v)) - psch[b]) for b in slack_buses
    )
    # Assert convergence and consistency with pandapower.
    assert it < max_iter, f"Load flow not converged in {it} iterations."
    assert np.allclose(
        v, net._ppc["internal"]["V"], atol=comp_tol, rtol=0
    ), f'Voltage\npp:\t\t{net._ppc["internal"]["V"]}\nsolved:\t{v}'
    assert np.allclose(
        p_slack, net.res_ext_grid["p_mw"].sum(), atol=comp_tol, rtol=0
    ), f'Slack Power\npp:\t\t{net.res_ext_grid["p_mw"].sum()}\nsolved:\t{p_slack}'
    return p_slack


def main():
    net = ppnw.case9()
    pp.runpp(net)  # Make sure network contains ybus and the solution values.

    print(net)
    p_slack = run_lf(net)
    print(f"Slack generator power output: {p_slack}")


if __name__ == "__main__":
    main()
