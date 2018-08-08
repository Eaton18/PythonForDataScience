
import logging

from os import path

# create two log class
LOG1 = logging.getLogger("a.b.c")
LOG2 = logging.getLogger("d.e")

# print("Current file:\t", path.dirname(path.dirname(path.dirname(__file__))))

# /Users/yigeng/Documents/workspace/data-science-learning/python/datasets/output/log_ops.log
output_log = path.dirname(path.dirname(path.dirname(__file__))) + "/datasets/output/log_ops.log"

# create handler object
console = logging.FileHandler(output_log,'a')

# log output format
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console.setFormatter(formatter)
# set filter
# can set muitl filters, the log info won't be output if not pass each filter
filter=logging.Filter('a.b')
console.addFilter(filter)

# binding handler object for two log class
LOG1.addHandler(console)
LOG2.addHandler(console)
# set log output level, there only output the log info which level > logging.INFO
LOG1.setLevel(logging.INFO)
LOG2.setLevel(logging.DEBUG)

# output some log info
LOG1.debug('debug')
LOG1.info('info')
LOG1.warning('warning')
LOG1.error('error')
LOG1.critical('critical')

LOG2.debug('debug')
LOG2.info('info')
LOG2.warning('warning')
LOG2.error('error')
LOG2.critical('critical')

