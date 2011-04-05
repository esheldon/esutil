from glob import glob

def configuration(parent_package='',top_path=None):
    from numpy.distutils.misc_util import Configuration

    config = Configuration('htm',parent_package,top_path)

    # note the glob occurs relative to base directory, but the
    # include dirs is relative to this dir
    sources = ['cgauleg.cc','cgauleg_wrap.cc']
    config.add_extension('_cgauleg', sources, include_dirs='../include')

    return config
