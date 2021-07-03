#
# BigNums.py
#
# Silly little utility to make big numbers so one can see the current iteration 
# number from a far distance... like while at the sink washing dishes while their
# computer is on the other end of the kitchen table.
#


NUM_HEIGHT = 6

NumberFont = {}

NumberFont[0] = [ '  ___  ',
                  ' / _ \ ',
                  '| | | |',
                  '| | | |',
                  '| |_| |',
                  ' \___/ '   ]
NumberFont[1] = [ ' __ ',
                  '/_ |',
                  ' | |',
                  ' | |',
                  ' | |',
                  ' |_|'      ]
NumberFont[2] = [ ' ___  ',
                  '|__ \ ',
                  '   ) |',
                  '  / / ',
                  ' / /_ ',
                  '|____|'    ]
NumberFont[3] = [ ' ____  ',
                  '|___ \ ',
                  '  __) |',
                  ' |__ < ',
                  ' ___) |',
                  '|____/ '   ]
NumberFont[4] = [ ' _  _   ',
                  '| || |  ',
                  '| || |_ ',
                  '|__   _|',
                  '   | |  ',
                  '   |_|  '  ]
NumberFont[5] = [ ' _____ ',
                  '| ____|',
                  '| |__  ',
                  '|___ \ ',
                  ' ___) |',
                  '|____/ '   ]
NumberFont[6] = [ '   __  ',
                  '  / /  ',
                  ' / /_  ',
                  "| '_ \ ",
                  '| (_) |',
                  ' \___/ '   ]
NumberFont[7] = [ ' ______ ',
                  '|____  |',
                  '    / / ',
                  '   / /  ',
                  '  / /   ',
                  ' /_/    '  ]
NumberFont[8] = [ '  ___  ',
                  ' / _ \ ',
                  '| (_) |',
                  ' > _ < ',
                  '| (_) |',
                  ' \___/ '   ]
NumberFont[9] = [ '  ___  ',
                  ' / _ \ ',
                  '| (_) |',
                  ' \__, |',
                  '   / / ',
                  '  /_/  '   ]


#
# IntToNumFont... converts an integer to a string of 'Big' numbers
# 
def IntToNumFont (num):

    outputStr = ''

    # convert the number into a list of separate digits
    nums = [int(d) for d in str(num)]

    for y in range (NUM_HEIGHT):
        for i in range(len(nums)):
            outputStr += NumberFont[ nums[i] ][y]

            if i == len(nums)-1:
                outputStr += '\n'
            else:
                outputStr += ' '

    return outputStr



