import copy
from dataclasses import dataclass
from logging import WARNING, Formatter, LogRecord, StreamHandler
from typing import Any, List


@dataclass
class ConsoleLogConfig:
    level: int = WARNING
    delimeter: str = ' '


class ConsoleLogger(StreamHandler):
    def __init__(self, config: ConsoleLogConfig):
        super().__init__()

        self.setFormatter(Formatter(
            fmt='[%(levelname)s] %(asctime)s "%(pathname)s:%(lineno)s" \n%(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        ))

        self.setLevel(config.level)
        self.dilimeter = config.delimeter

    def emit(self, record: LogRecord):
        try:
            recordTemp = copy.deepcopy(record)
            msgs: List[Any] = record.msg
            recordTemp.msg = self.dilimeter.join([str(m) for m in msgs])
        except Exception:
            self.handleError(record)
    
        super().emit(recordTemp)
