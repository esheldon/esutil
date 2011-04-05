from glob import glob

def configuration(parent_package='',top_path=None):
    from numpy.distutils.misc_util import Configuration

    config = Configuration('htm',parent_package,top_path)

    # note the glob occurs relative to base directory, but the
    # include dirs is relative to this dir
    sources = glob('esutil/htm/htm_src/*.cpp') + glob('esutil/htm/*.cc')
    config.add_extension('_htmc', sources, include_dirs=['htm_src','../include'])

    return config
