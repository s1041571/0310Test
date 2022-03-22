from .ConsoleLogger import ConsoleLogConfig, ConsoleLogger
# from .DBLogger import *
from .FileLogger import FileLogConfig, FileLogger


def console_logger(config: ConsoleLogConfig) -> ConsoleLogger:
    return ConsoleLogger(config)

def file_logger(config: FileLogConfig) -> FileLogger:
    return FileLogger(config)
'''
def db_logger(config: DBLogConfig) -> MariaLogger:
    if config.dbType == DBType.Null or config.dbType == DBType.CSV:
        return CsvLogger(config)
    elif config.dbType == DBType.Maria:
        return MariaLogger(config)
    return
'''