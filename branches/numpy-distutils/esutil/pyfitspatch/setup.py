from glob import glob

def configuration(parent_package='',top_path=None):
    from numpy.distutils.misc_util import Configuration

    config = Configuration('pyfitspatch',parent_package,top_path)

    # note the glob occurs relative to base directory, but the
    # include dirs is relative to this dir
    sources = glob('esutil/pyfitspatch/*.c')
    config.add_extension('pyfitsComp', sources)

    return config
