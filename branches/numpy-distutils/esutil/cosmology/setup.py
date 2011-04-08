from glob import glob

def configuration(parent_package='',top_path=None):
    from numpy.distutils.misc_util import Configuration

    config = Configuration('cosmology',parent_package,top_path)

    sources = ['gauleg.c','cosmolib.c','cosmolib_wrap.c']
    config.add_extension('_cosmolib', sources)

    return config
