try:
    import imageio_ffmpeg
    import matplotlib as mpl
    import pytest
    @pytest.fixture(autouse=True)
    def define_ffmpeg(doctest_namespace):
        mpl.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
        doctest_namespace["mpl.rcParams"] = mpl.rcParams
except ModuleNotFoundError:
    pass
