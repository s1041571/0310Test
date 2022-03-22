'''
develop by: MMFAA1 JoeJHHuang (ext: 7510)
'''
import logging
import sys
from typing import Any
import os
from . import LoggerFunc
from .ConsoleLogger import ConsoleLogConfig
# from .DBLogger import DBLogConfig, DBType
from .FileLogger import FileLogConfig

__all__ = [
    "LogLevel", "LogType", "Logger",
    "ConsoleLogConfig", "FileLogConfig",
    # "DBLogConfig", "DBType",
    ]

class LogLevel:
    DEBUG = logging.DEBUG
    INFO = logging.INFO
    WARNING = logging.WARNING
    ERROR = logging.ERROR

class LogType:
    Console: int = 0b0001
    File: int    = 0b0010
    DB: int      = 0b0100

class Logger:
    __logger: logging.Logger = logging.Logger(__name__)

    @classmethod
    def config(
        cls,
        logTypes: int=0,
        consoleLogConfig: ConsoleLogConfig = None,
        fileLogConfig: FileLogConfig = None,
        # dbLogConfig: DBLogConfig = None,
        ):
        '''
        Basic settings of logger

        Args:
            logTypes (int): set log types, ex: LogType.Console | LogType.File | LogType.DB
            consoleLogConfig (ConsoleLogConfig): config of console log
            fileLogConfig (FileLogConfig): config of file log, create new file every day
            dbLogConfig (DBLogConfig): config of DB log, DB support: mariadb(mysql), postgres, mongodb, mssql, sqlite3
        '''
        if logTypes & LogType.Console:
            if consoleLogConfig == None:
                print("WARNING: console log enable but console log config not set, console log will not work.")
            else:
                consoleLogger = LoggerFunc.console_logger(consoleLogConfig)
                cls.__logger.addHandler(consoleLogger)

        if logTypes & LogType.File:
            if fileLogConfig == None:
                print("WARNING: file log enable but file log config not set, file log will not work.")
            else:
                fileLogger = LoggerFunc.file_logger(fileLogConfig)
                cls.__logger.addHandler(fileLogger)
        '''
        if logTypes & LogType.DB:
            if dbLogConfig == None:
                print("WARNING: DB log enable but DB log config not set, DB log will not work.")
            else:
                dbLogger = LoggerFunc.db_logger(dbLogConfig)
                cls.__logger.addHandler(dbLogger)
        '''

    @classmethod
    def log(cls, level: int, *msgs: Any):
        # Changed in version 3.8: The stacklevel parameter was added.
        if sys.version_info.major == 3 and sys.version_info.minor >= 8:
            cls.__logger.log(level, msgs, stacklevel=3)
        else:
            import inspect
            stack = inspect.stack()[2]
            cls.__logger.log(level, msgs, extra={
                'stack_filename': stack.filename,
                'stack_lineno': stack.lineno,
                'stack_function': stack.function,
            })

    @classmethod
    def debug(cls, *msgs: Any):
        cls.log(LogLevel.DEBUG, *msgs)

    @classmethod
    def info(cls, *msgs: Any):
        cls.log(LogLevel.INFO, *msgs)

    @classmethod
    def warning(cls, *msgs: Any):
        cls.log(LogLevel.WARNING, *msgs)

    @classmethod
    def error(cls, *msgs: Any):
        cls.log(LogLevel.ERROR, *msgs)
