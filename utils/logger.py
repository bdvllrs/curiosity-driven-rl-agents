from os.path import abspath
from datetime import datetime
from inspect import getframeinfo, stack
from utils.singleton import Singleton


class Logger:
    def __init__(self):
        self.messages = []
        self.file = None
        self.verbose = False

    def set(self, **params):
        if "file" in params.keys():
            self.file = params['file']
        if "verbose" in params.keys():
            self.verbose = params["verbose"]

    def log(self, message, alert_type=""):
        caller = getframeinfo(stack()[1][0])
        self.messages.append({
            "time": datetime.today().strftime('%Y-%m-%d %H:%M:%S'),
            "message": message,
            "stack": f"{caller.filename}:L {caller.lineno}",
            "type": alert_type
        })
        if self.verbose:
            print(self._to_string(self.messages[-1]))
        if self.file is not None:
            with open(self.file, "a+") as f:
                f.write(self._to_string(self.messages[-1]) + "\n")

    def error(self, message):
        self.log(message, "error")

    def warning(self, message):
        self.log(message, "warning")

    def _to_string(self, msg):
        message = ""
        if msg['type'] != '':
            message += f"*{msg['type']}* "
        message += f"[{msg['time']}] {msg['stack']} - {msg['message']}"
        return message

    def _get_fulltext(self):
        message = ""
        for msg in self.messages:
            message += self._to_string(msg) + "\n"
        return message

    def dump(self, file=None):
        file = self.file if file is None else file
        file = abspath(file)
        with open(file, "w") as f:
            f.write(self._get_fulltext())


logger = Singleton(Logger)
