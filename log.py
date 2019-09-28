LOGGING_CONF = {
    'version': 1,
    'formatters': {
        'simple': {
            'format': '%(levelname)s %(asctime)s %(module)s %(thread)d: %(message)s'
        },
    },
    'handlers': {
        'console': {
            'level': 'DEBUG',
            'class': 'logging.StreamHandler',
            'formatter': 'simple'
        },
    },
    'loggers': {
        'main': {
            'handlers': ['console'],
            'level': 'INFO',
        },
        'db': {
            'handlers': ['console'],
            'level': 'DEBUG',
        }
    }
}

