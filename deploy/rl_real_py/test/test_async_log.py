import csv
import queue
import tempfile
import threading
from pathlib import Path

from rl_real_py.rl_real_common import RL_real


def test_async_log_writer():
    with tempfile.TemporaryDirectory() as tmp_dir:
        path = Path(tmp_dir) / "log.csv"
        rows = queue.SimpleQueue()
        worker = threading.Thread(target=RL_real._write_log, args=(path, ["a", "b"], rows))
        worker.start()
        rows.put([1, 2])
        rows.put([3, 4])
        rows.put(None)
        worker.join(2)

        assert not worker.is_alive()
        with path.open() as log_f:
            assert list(csv.reader(log_f)) == [["a", "b"], ["1", "2"], ["3", "4"]]
