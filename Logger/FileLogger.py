import copy
import os
from dataclasses import dataclass
from logging import WARNING, Formatter, LogRecord
from logging.handlers import TimedRotatingFileHandler
from typing import Any, List


@dataclass
class FileLogConfig:
    level: int = WARNING
    delimeter: str = ' '
    newline: bool = True

    dirname: str=''
    suffix: str=''

    def filename(self):
        pathname = self.dirname
        if pathname != '' and pathname[0] != '.':
            pathname = os.path.join('.', pathname)
        return os.path.join(pathname, self.suffix+".log")

class FileLogger(TimedRotatingFileHandler):
    def __init__(self, config: FileLogConfig):
        if config.dirname != '' and not os.path.exists(config.dirname):
            os.mkdir(config.dirname)
            
        super().__init__(filename=config.filename(), when='D')

        self.namer = self.__namer
        self.suffix = "%Y%m%d"
        self.extMatch = r"^\d{4}\d{2}\d{2}(\.\w+)?$"

        lf = '\n' if config.newline else ' - ' # line feed
        self.setFormatter(Formatter(
            fmt='[%(levelname)s] %(asctime)s "%(pathname)s:%(lineno)s"' + lf + '%(message)s',
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

    def __namer(self, defaultName: str):
        base_filename, _, date = defaultName.split('.')

        sep = '-'
        if base_filename[-1] in ['/', '\\']:
            sep = ''

        return f"{base_filename}{sep}{date}.log"
