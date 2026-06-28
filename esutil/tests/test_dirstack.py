def test_dirstack_context():
    import tempfile
    import os
    from os.path import abspath
    from esutil.ostools import DirStack

    with tempfile.TemporaryDirectory() as tmpdir:

        # try abspath to get tests to work on macos
        orig_dir = os.getcwd()

        with DirStack() as ds:
            ds.push(tmpdir)
            assert abspath(os.getcwd()) == abspath(tmpdir)

        assert abspath(os.getcwd()) == abspath(orig_dir)


def test_dirstack_nocontext():
    import tempfile
    import os
    from os.path import abspath
    from esutil.ostools import DirStack

    with tempfile.TemporaryDirectory() as tmpdir:

        # try abspath to get tests to work on macos
        orig_dir = os.getcwd()
        ds = DirStack()
        ds.push(tmpdir)
        assert abspath(os.getcwd()) == abspath(tmpdir)
        ds.pop()

        assert abspath(os.getcwd()) == abspath(orig_dir)
