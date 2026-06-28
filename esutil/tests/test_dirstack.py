def test_dirstack_context():
    import tempfile
    import os
    from esutil.ostools import DirStack

    with tempfile.TemporaryDirectory() as tmpdir:

        # on macos CI the actual directory has a different name
        # than tmpdir!  Do a first run to get that name
        orig_dir = os.getcwd()
        os.chdir(tmpdir)
        tmpdir_actual_name = os.getcwd()
        os.chdir(orig_dir)

        with DirStack() as ds:
            ds.push(tmpdir)
            assert os.getcwd() == tmpdir_actual_name

        assert os.getcwd() == orig_dir


def test_dirstack_nocontext():
    import tempfile
    import os
    from esutil.ostools import DirStack

    with tempfile.TemporaryDirectory() as tmpdir:

        # on macos CI the actual directory has a different name
        # than tmpdir!  Do a first run to get that name

        orig_dir = os.getcwd()
        os.chdir(tmpdir)
        tmpdir_actual_name = os.getcwd()
        os.chdir(orig_dir)

        ds = DirStack()
        ds.push(tmpdir)
        assert os.getcwd() == tmpdir_actual_name
        ds.pop()

        assert os.getcwd() == orig_dir
