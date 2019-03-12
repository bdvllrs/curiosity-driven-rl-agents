from os.path import abspath
from datetime import datetime
from inspect import getframeinfo, stack
from utils.utils import Singleton


class Logger:
    def __init__(self):
        self.messages = []

    def log(self, message, alert_type=""):
        caller = getframeinfo(stack()[1][0])
        self.messages.append({
            "time": datetime.today().strftime('%Y-%m-%d %H:%M:%S'),
            "message": message,
            "stack": f"{caller.filename}:L {caller.lineno}",
            "type": alert_type
        })

    def error(self, message):
        self.log(message, "error")

    def warning(self, message):
        self.log(message, "warning")

    def _to_string(self):
        message = ""
        for msg in self.messages:
            if msg['type'] != '':
                message += f"*{msg['type']}* "
            message += f"[{msg['time']}] {msg['stack']} - {msg['message']} \n"
        return message

    def dump(self, file):
        file = abspath(file)
        with open(file, "w") as f:
            f.write(self._to_string())


logger = Singleton(Logger)
