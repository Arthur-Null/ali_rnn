import logging

logging.basicConfig(level=logging.DEBUG,
        format='%(asctime)s line:%(lineno)d %(levelname)s:  %(message)s ',
        datefmt='%Y-%m-%d %H:%M:%S')

def print_log(msg, type='info'):
    if not type in ['info', 'warning', 'debug']:
        raise NotImplementedError('The type of log can not be {0}'.format(type))
    if type == 'info':
        logging.info(msg)
    elif type == 'warning':
        logging.warning(msg)
    elif type == 'debug':
        logging.debug(msg)