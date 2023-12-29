import pooch

"""
Large files that are downloaded/cached automagically using the pooch library.

  How to add new entries to this list:

  Get expected hash by running md5sum or md5 on the file
  For Dropbox links, use "Copy Link" to get the URL,
    replace "www.dropbox.com" with "dl.dropboxusercontent.com",
    replace "dl=0" with "dl=1"
"""

files = {
    "aerial_1970_clipped": {
        "url": "https://dl.dropboxusercontent.com/scl/fi/avl01m8xt33m32jhfrqup/Aerial1970_clipped.tif?rlkey=bap65eirnwwk8qqkedinfxnuy&dl=1",
        "hash": "md5:9ac6601f5f9e8160efebabe6f26c3657",
    },
    "aerial_1970": {
        "url": "https://dl.dropboxusercontent.com/scl/fi/u41iaog9j42yrtyiulhjr/Aerial1970_May12_2021_secondGCPsTest.tif?rlkey=slbap8hywsa5hlzc65q553yl8&dl=1",
        "hash": "md5:55601074d9a60e4cb10e779a06d58f5c",
    }
}


def get_file(which):
    assert which in files, f"Unknown file {which}"
    file = files[which]
    return pooch.retrieve(url=file["url"], known_hash=file["hash"])