from PyQt5.QtCore import QThread, pyqtSignal
import logging

logger = logging.getLogger(__name__)

class OperationCancelledError(Exception):
    """Thrown when user press cancel."""
    pass

class Worker(QThread):
    progress_signal = pyqtSignal(int, str)
    finished_signal = pyqtSignal(object)
    error_signal = pyqtSignal(str)
    cancel_signal = pyqtSignal()

    def __init__(self, func, *args, **kwargs):
        super().__init__()
        self.func = func
        self.args = args
        self.kwargs = kwargs

    def run(self):
        hooks = {
            "on_progress": lambda val, msg: self.progress_signal.emit(val, msg),
            "check_interruption": lambda: self.isInterruptionRequested()
            }
        try:
            result = self.func(
                *self.args,
                hooks=hooks,
                **self.kwargs
                )

        except OperationCancelledError:
            self.cancel_signal.emit()

        except  Exception as e:
            self.error_signal.emit(str(e))
            logger.error(f"An error occured: {str(e)}")

        else:
            self.finished_signal.emit(result)

class WorkerHelper:
    def __init__(self, hooks=None):
        self.hooks = hooks or {}

    def progress(self, val, msg):
        cb = self.hooks.get("on_progress")
        if cb:
            cb(val, msg)

    def cancelled(self):
        cb = self.hooks.get("check_interruption")
        return cb() if cb else False
    
    def check_cancel(self):
        if self.cancelled():
            raise OperationCancelledError("User cancelled")


        