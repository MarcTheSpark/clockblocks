import threading
import time

true_sleep = time.sleep
true_time = time.time


def get_compressed_sleep(compression_factor):
    def mock_sleep(dt):
        true_sleep(dt/compression_factor)
    return mock_sleep


def get_compressed_time(compression_factor):
    start_time = true_time()

    def mock_time():
        return (true_time() - start_time) * compression_factor + start_time
    return mock_time


def get_compressed_event(compression_factor):
    class MockEvent(threading.Event):
        def wait(self, timeout=None) -> bool:
            # print(f"CALLED MOCK wait {threading.currentThread()}, {timeout}")
            return super().wait(timeout/compression_factor if timeout is not None else timeout)
    return MockEvent
