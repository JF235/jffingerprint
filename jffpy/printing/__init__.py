def format_time(seconds: float) -> str:
    """
    Converts a given value in seconds to a human-readable string in hours, minutes, seconds,
    milliseconds, microseconds, or nanoseconds based on the scale.

    :param seconds: The time value in seconds.
    :type seconds: float
    :return: A string representing the time value in the most appropriate unit.
    :rtype: str
    """
    if seconds >= 3600:
        return f"{seconds / 3600:.2f} h"
    elif seconds >= 60:
        return f"{seconds / 60:.2f} min"
    elif seconds >= 1:
        return f"{seconds:.2f} s"
    elif seconds >= 1e-3:
        return f"{seconds * 1e3:.2f} ms"
    elif seconds >= 1e-6:
        return f"{seconds * 1e6:.2f} us"
    else:
        return f"{seconds * 1e9:.2f} ns"

def purify_name(name):
    # 006022331520_E25916112409340610_dedo3.tpt
    # -> 31520_40610_d3
    if '-' in name:
        return name.split(".")[0].rjust(15)
    else:
        parts = name.split("_")
        frankenstein = parts[0][-5:] + "_" + parts[1][-5:] + "_d"
        fingerno = parts[2].split("dedo")[1].split(".")[0]
        if int(fingerno) < 10:
            frankenstein += "0"
        frankenstein += fingerno
        return frankenstein.rjust(15)