import re

from ollama.monitoring.profiling import ProfilerManager


def busy_work(n: int) -> int:
    s = 0
    for i in range(n):
        s += i * i
    return s


def test_profiler_start_stop_and_stats() -> None:
    mgr = ProfilerManager()
    assert not mgr.is_running()
    mgr.start(enable_tracemalloc=False)
    assert mgr.is_running()

    # perform some CPU work
    busy_work(5000)

    stats = mgr.stop_and_get_stats(sort_by="cumulative", top_n=10)
    assert isinstance(stats, str)
    # Expect the busy_work function to appear in the stats output
    assert re.search(r"busy_work", stats) is not None
    assert not mgr.is_running()
