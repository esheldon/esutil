from glob import glob

def configuration(parent_package='',top_path=None):
    from numpy.distutils.misc_util import Configuration

    config = Configuration('recfile',parent_package,top_path)

    # note the glob occurs relative to base directory, but the
    # include dirs is relative to this dir
    sources = ['records.cpp','records_wrap.cpp']
    config.add_extension('_records', sources, include_dirs='../include')

    return config
