from pyomo.environ import floor

from src.base import Case


def tzl_constraint_1(m, w, t):  # type:ignore
    return m.tzl_base[t] - m.delta <= m.tzl[w, t]


def tzl_constraint_2(m, w, t):  # type:ignore
    return m.tzl_base[t] + m.delta >= m.tzl[w, t]


def tzu_constraint_1(m, w, t):  # type:ignore
    return m.tzu_base[t] - m.delta <= m.tzu[w, t]


def tzu_constraint_2(m, w, t):  # type:ignore
    return m.tzu_base[t] + m.delta >= m.tzu[w, t]


def twl_constraint_1(m, w, t):  # type:ignore
    return m.twl_base[t] - m.delta <= m.twl[w, t]


def twl_constraint_2(m, w, t):  # type:ignore
    return m.twl_base[t] + m.delta >= m.twl[w, t]


def twu_constraint_1(m, w, t):  # type:ignore
    return m.twu_base[t] - m.delta <= m.twu[w, t]


def twu_constraint_2(m, w, t):  # type:ignore
    return m.twu_base[t] + m.delta >= m.twu[w, t]


def delta_constraint(m):  # type:ignore
    return m.delta <= m.delta_max


def tzu_baseline_constraint(m, t):  # type:ignore
    if t > 0:
        return m.tzu_base[t] == m.tzu_base[t - 1] + 1 / m.Czu * m.dt * (
            1 / m.Rzuzl * (m.tzl_base[t - 1] - m.tzu_base[t - 1])
            + 1 / m.Rwz * (m.twu_base[t - 1] - m.tzu_base[t - 1])
        )
    else:
        return m.tzu_base[0] == m.setpoint_uz


def tzl_baseline_constraint(m, t):  # type:ignore
    if t > 0:
        return m.tzl_base[t] == m.tzl_base[t - 1] + 1 / m.Czl * m.dt * (
            1 / m.Rzuzl * (m.tzu_base[t - 1] - m.tzl_base[t - 1])
            + 1 / m.Rwz * (m.twl_base[t - 1] - m.tzl_base[t - 1])
        )
    else:
        return m.tzl_base[0] == m.setpoint_lz


def twu_baseline_constraint(m, t):  # type:ignore
    if t > 0:
        # h = t // self.hour_steps  # only one power step per hour
        h = floor(t / m.hour_steps)
        return m.twu_base[t] == m.twu_base[t - 1] + 1 / m.Cwu * m.dt * (
            (1 - m.regime[t]) * 1 / m.Rwua1 * (m.ta - m.twu_base[t - 1])
            + m.regime[t] * 1 / m.Rwua2 * (m.ta - m.twu_base[t - 1])
            + 1 / m.Rww * (m.twl_base[t - 1] - m.twu_base[t - 1])
            + 1 / m.Rwz * (m.tzu_base[t - 1] - m.twu_base[t - 1])
            + m.p_base_uz[h]
        )
    else:
        return m.twu_base[0] == m.setpoint_uz


def twl_baseline_constraint(m, t):  # type:ignore
    if t > 0:
        # h = t // self.hour_steps  # only one power step per hour
        h = floor(t / m.hour_steps)  # only one power step per hour
        return m.twl_base[t] == m.twl_base[t - 1] + 1 / m.Cwl * m.dt * (
            1 / m.Rwla1 * (m.ta - m.twl_base[t - 1])
            + 1 / m.Rww * (m.twu_base[t - 1] - m.twl_base[t - 1])
            + 1 / m.Rwz * (m.tzl_base[t - 1] - m.twl_base[t - 1])
            + m.p_base_lz[h]
        )
    else:
        return m.twl_base[0] == m.setpoint_lz


def tzu_constraint(m, w, t):  # type:ignore
    if t > 0:
        return m.tzu[w, t] == m.tzu[w, t - 1] + 1 / m.Czu * m.dt * (
            1 / m.Rzuzl * (m.tzl[w, t - 1] - m.tzu[w, t - 1])
            + 1 / m.Rwz * (m.twu[w, t - 1] - m.tzu[w, t - 1])
        )
    else:
        return m.tzu[w, 0] == m.setpoint_uz


def tzl_constraint(m, w, t):  # type:ignore
    if t > 0:
        return m.tzl[w, t] == m.tzl[w, t - 1] + 1 / m.Czl * m.dt * (
            1 / m.Rzuzl * (m.tzu[w, t - 1] - m.tzl[w, t - 1])
            + 1 / m.Rwz * (m.twl[w, t - 1] - m.tzl[w, t - 1])
        )
    else:
        return m.tzl[w, 0] == m.setpoint_lz


def twu_constraint(m, w, t):  # type:ignore
    if t > 0:
        # h = t // self.hour_steps  # only one power step per hour
        h = floor(t / m.hour_steps) if m.name != Case.FCR.name else t
        return m.twu[w, t] == m.twu[w, t - 1] + 1 / m.Cwu * m.dt * (
            (1 - m.regime[t]) * 1 / m.Rwua1 * (m.ta - m.twu[w, t - 1])
            + m.regime[t] * 1 / m.Rwua2 * (m.ta - m.twu[w, t - 1])
            + 1 / m.Rww * (m.twl[w, t - 1] - m.twu[w, t - 1])
            + 1 / m.Rwz * (m.tzu[w, t - 1] - m.twu[w, t - 1])
            + m.pt_uz[w, h]
        )
    else:
        return m.twu[w, 0] == m.setpoint_uz


def twl_constraint(m, w, t):  # type:ignore
    if t > 0:
        # h = t // self.hour_steps  # only one power step per hour
        h = floor(t / m.hour_steps) if m.name != Case.FCR.name else t
        return m.twl[w, t] == m.twl[w, t - 1] + 1 / m.Cwl * m.dt * (
            1 / m.Rwla1 * (m.ta - m.twl[w, t - 1])
            + 1 / m.Rww * (m.twu[w, t - 1] - m.twl[w, t - 1])
            + 1 / m.Rwz * (m.tzl[w, t - 1] - m.twl[w, t - 1])
            + m.pt_lz[w, h]
        )
    else:
        return m.twl[w, 0] == m.setpoint_lz


def boundary_constraint1(m, w):  # type:ignore
    # temperature at the end must equal tmeperature for the beginning (or data)
    return m.tzl[w, m._n_steps - 1] >= m.tzl_base[m._n_steps - 1] - 0.1


def boundary_constraint2(m, w):  # type:ignore
    # temperature at the end must equal tmeperature for the beginning (or data)
    return m.tzu[w, m._n_steps - 1] >= m.tzu_base[m._n_steps - 1] - 0.1
