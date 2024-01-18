import pooch

"""
Large files that are downloaded/cached automagically using the pooch library.

  How to add new entries to this list:

  Get expected hash by running md5sum or md5 on the file
  For Dropbox links, use "Copy Link" to get the URL,
    replace "www.dropbox.com" with "dl.dropboxusercontent.com",
    replace "dl=0" with "dl=1"
"""

ONDEMAND = pooch.create(
    path=pooch.os_cache("deepmap"),
    base_url="https://dl.dropboxusercontent.com/scl/fi",
    registry={
        "aerial_1970_clipped": "md5:9ac6601f5f9e8160efebabe6f26c3657",
        "aerial_1970": "md5:55601074d9a60e4cb10e779a06d58f5c",
        "neon": "md5:a212563fdff0ed211ee1014b0828846b",
    },
    urls={
        "aerial_1970_clipped": "https://dl.dropboxusercontent.com/scl/fi/avl01m8xt33m32jhfrqup/Aerial1970_clipped.tif?rlkey=bap65eirnwwk8qqkedinfxnuy&dl=1",
        "aerial_1970": "https://dl.dropboxusercontent.com/scl/fi/u41iaog9j42yrtyiulhjr/Aerial1970_May12_2021_secondGCPsTest.tif?rlkey=slbap8hywsa5hlzc65q553yl8&dl=1",
        "neon": "https://dl.dropboxusercontent.com/scl/fi/zu1k367yli6bwc0ai2egb/neon.pt?rlkey=m6s25w18hapzfesyf5ut9kift&dl=1"
    },
    env="DEEPMAP_POOCH_DIR",
)


def get_file(which):
    return ONDEMAND.fetch(which)


def get_all():
    """
    Download all available files. Slow, but useful for one time cacheing.
    """
    return [get_file(f) for f in ONDEMAND.registry]
