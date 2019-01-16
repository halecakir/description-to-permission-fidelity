import logging
from logging.config import dictConfig

logging_config = dict(
    disable_existing_loggers = True,
    version = 1,
    formatters = {
        'f': {'format':
              '%(asctime)s %(name)-12s %(levelname)-8s %(message)s'}
        },
    handlers = {
        'stream': {'class': 'logging.StreamHandler',
              'formatter': 'f',
              'level': logging.INFO},
        'file': {
            'level': logging.INFO,
            'formatter': 'f',
            'class': 'logging.FileHandler',
            'filename': 'out.log',
            'mode': 'w',
            }
        },
    root = {
        'handlers': ['file'],
        'level': logging.INFO,
        },
)

dictConfig(logging_config)
logger = logging.getLogger('project')
logger.propagate = False