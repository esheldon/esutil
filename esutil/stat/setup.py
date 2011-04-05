def configuration(parent_package='',top_path=None):
    from numpy.distutils.misc_util import Configuration

    config = Configuration('stat',parent_package,top_path)
    sources = ['chist.cc','chist_wrap.cc']
    config.add_extension('_chist', sources, include_dirs='../include')

    return config

