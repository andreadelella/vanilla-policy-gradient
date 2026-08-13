import numpy as np


# Two-sided 95% Student-t critical values for 1..30 degrees of freedom.
# Keeping the small-sample values local avoids adding SciPy solely for a
# single quantile lookup. For larger samples, the Cornish-Fisher expansion
# below rapidly approaches the standard-normal critical value.
_T_CRITICAL_95 = (
    None,
    12.7062047364,
    4.3026527297,
    3.1824463053,
    2.7764451052,
    2.5705818356,
    2.4469118511,
    2.3646242510,
    2.3060041352,
    2.2621571629,
    2.2281388520,
    2.2009851601,
    2.1788128297,
    2.1603686565,
    2.1447866879,
    2.1314495456,
    2.1199052992,
    2.1098155778,
    2.1009220402,
    2.0930240544,
    2.0859634473,
    2.0796138447,
    2.0738730679,
    2.0686576104,
    2.0638985616,
    2.0595385528,
    2.0555294386,
    2.0518305165,
    2.0484071418,
    2.0452296421,
)


def student_t_critical_95(n_samples):
    """Return the two-sided 95% Student-t critical value for a sample mean."""
    if n_samples < 2:
        raise ValueError("At least two independent samples are required for a confidence interval")

    degrees_of_freedom = n_samples - 1

    if degrees_of_freedom <= 30:
        return _T_CRITICAL_95[degrees_of_freedom]

    # With many samples, Student-t approaches the normal value 1.96.
    # This formula gives a close value without requiring SciPy.
    z = 1.959963984540054
    df = float(degrees_of_freedom)
    return (
        z
        + (z**3 + z) / (4.0 * df)
        + (5.0 * z**5 + 16.0 * z**3 + 3.0 * z) / (96.0 * df**2)
        + (3.0 * z**7 + 19.0 * z**5 + 17.0 * z**3 - 15.0 * z)
        / (384.0 * df**3)
    )


def mean_confidence_interval(data):
    """Mean and two-sided 95% Student-t interval across independent samples."""
    data = np.asarray(data, dtype=np.float64)
    if data.ndim == 0 or data.shape[0] < 2:
        raise ValueError("At least two independent samples are required for a confidence interval")

    mean = data.mean(axis=0)
    sem = data.std(axis=0, ddof=1) / np.sqrt(data.shape[0])
    margin = student_t_critical_95(data.shape[0]) * sem
    return mean, mean - margin, mean + margin
