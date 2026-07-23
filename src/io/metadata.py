import numpy as np


def parse_metadata(header):
    us = header["USHEADER"][0, 0]

    return {
        "system": str(us["system"][0]),
        "transducer": str(us["transducer"][0]),
        "signal_type": str(us["signalType"][0]),
        "c": float(us["c"][0, 0]),
        "fc": float(us["fc"][0, 0]),
        "fs": float(us["fs"][0, 0]),
        "pitch": float(us["pitch"][0, 0]),
        "angles": np.squeeze(us["xmitAngles"]),
    }