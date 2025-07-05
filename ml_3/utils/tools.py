import logging

class CustomFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        if "ml-3" in record.pathname:
            module = "ML-3"
        
        return "[{} - {}] [{} {}: {}] {}".format(
            module,
            record.levelname,
            self.formatTime(record, datefmt="%B-%d %H:%M:%S"),
            record.filename,
            record.lineno,
            record.getMessage()
        )