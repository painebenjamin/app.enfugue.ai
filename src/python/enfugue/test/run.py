import sys
import os
import traceback
import multiprocessing

from pibble.util.log import logger, DebugUnifiedLoggingContext
from pibble.util.helpers import resolve

HERE = os.path.dirname(os.path.abspath(__file__))


class TestThread(multiprocessing.Process):
    def __init__(self, queue, test_module_name):
        super(TestThread, self).__init__()
        self.queue = queue
        self.test_module_name = test_module_name

    def run(self):
        try:
            test_module = __import__(
                "enfugue.test.{0}".format(self.test_module_name), fromlist=["enfugue.test"]
            )
            result = test_module.main()
        except Exception as ex:
            self.queue.put(type(ex).__name__)
            self.queue.put(str(ex))


def run_test(test_file):
    queue = multiprocessing.Queue()
    test_module_name = os.path.splitext(os.path.basename(test_file))[0]
    test_thread = TestThread(queue, test_module_name)
    test_thread.start()
    test_thread.join()
    if not queue.empty():
        try:
            exception_type = resolve(queue.get())
        except:
            exception_type = Exception
        raise exception_type(queue.get())
    return test_module_name


def main():
    with DebugUnifiedLoggingContext():
        tests = {}
        for filename in os.listdir(HERE):
            if filename.endswith(".py"):
                try:
                    priority = int(filename[0])
                    if priority not in tests:
                        tests[priority] = []
                    tests[priority].append(filename)
                except ValueError:
                    pass
        logger.info(
            "Running {0} tests.".format(
                sum([len(tests[priority]) for priority in tests if priority > 0])
            )
        )
        ran = []
        for priority in sorted(tests.keys()):
            if priority > 0:
                for test in tests[priority]:
                    try:
                        logger.warning(
                            "Running test {0}".format(
                                os.path.splitext(os.path.basename(test))[0]
                            )
                        )
                        ran.append(run_test(test))
                    except Exception as ex:
                        logger.critical(
                            "Test {0} failed, final exception was: {1}({2})".format(
                                os.path.splitext(os.path.basename(test))[0],
                                type(ex).__name__,
                                str(ex),
                            )
                        )
                        logger.error(traceback.format_exc())
                        return 5
        logger.info("Successfully ran {0} tests: {1}".format(len(ran), ", ".join(ran)))
    return 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:3]))
