def test_dirstack_context():
    import tempfile
    import os
    from esutil.ostools import DirStack

    with tempfile.TemporaryDirectory() as tmpdir:
        orig_dir = os.getcwd()
        with DirStack() as ds:
            ds.push(tmpdir)
            assert os.getcwd() == tmpdir

        assert os.getcwd() == orig_dir


def test_dirstack_nocontext():
    import tempfile
    import os
    from esutil.ostools import DirStack

    with tempfile.TemporaryDirectory() as tmpdir:
        orig_dir = os.getcwd()
        ds = DirStack()
        ds.push(tmpdir)
        assert os.getcwd() == tmpdir
        ds.pop()

        assert os.getcwd() == orig_dir
